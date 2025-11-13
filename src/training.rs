use candle_core::Var;
use candle_nn::VarMap;
use log::{debug, info, warn};
use optimisers::lbfgs::{Lbfgs, ParamsLBFGS};
use optimisers::LossOptimizer;

use crate::SimpleModel;

pub(super) fn run_lbfgs_training<M: SimpleModel>(
    model: &M,
    varmap: &VarMap,
    params: ParamsLBFGS,
    lbfgs_steps: usize,
) -> anyhow::Result<f64> {
    let mut loss = model.loss()?;
    info!(
        "initial loss: {}",
        loss.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?
    );

    // create an optimiser
    let mut optimiser = Lbfgs::new(varmap.all_vars(), params, model)?;
    let mut fn_evals = 1;
    let mut converged = false;

    for step in 0..lbfgs_steps {
        // step the tensors by backpropagating the loss
        let res = optimiser.backward_step(&loss)?;
        match res {
            optimisers::ModelOutcome::Converged(new_loss, evals) => {
                info!("step: {}", step);
                info!(
                    "loss: {}",
                    new_loss
                        .to_dtype(candle_core::DType::F64)?
                        .to_scalar::<f64>()?
                );
                info!("test metric: {}", model.test_eval()?);
                fn_evals += evals;
                loss = new_loss;
                converged = true;
                info!("converged after {} fn evals", fn_evals);
                break;
            }
            optimisers::ModelOutcome::Stepped(new_loss, evals) => {
                debug!("step: {}", step);
                debug!(
                    "loss: {}",
                    new_loss
                        .to_dtype(candle_core::DType::F32)?
                        .to_scalar::<f32>()?
                );
                debug!("test acc: {:5.2}", model.test_eval()?);
                fn_evals += evals;
                loss = new_loss;
            }
        }
    }
    if !converged {
        info!("test acc: {:5.2}", model.test_eval()?);
        warn!("did not converge after {} fn evals", fn_evals);
    }
    info!(
        "loss: {}",
        loss.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?
    );
    info!("{} fn evals", fn_evals);
    Ok(loss.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?)
}

pub(super) fn l2_norm(vs: &[Var]) -> candle_core::Result<f64> {
    let mut norm = 0.;
    for v in vs {
        norm += v
            .as_tensor()
            .powf(2.)?
            .sum_all()?
            .to_dtype(candle_core::DType::F64)?
            .to_scalar::<f64>()?;
    }
    Ok(norm)
}
