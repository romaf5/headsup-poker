"""External-sampling SD-CFR on FHP with the authors' public code (DREAM repository, FHP_ES.py), in one process: the
reference run scored by docs/analysis/fhp_reference_br.py.

    REF_DEVICE=cuda PYTHONPATH=<DREAM>:<stubs> python docs/analysis/fhp_reference_run.py <iterations> <name> <out dir> \
        <traversals per player> <SGD steps> <batch> <advantage memory size>      # the paper: 10000 4000 10000, memory 4e7
"""
import os
import sys

from DREAM_and_DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DREAM_and_DeepCFR.TrainingProfile import TrainingProfile
from DREAM_and_DeepCFR.workers.driver.Driver import Driver
from PokerRL.game.games import Flop3Holdem

if __name__ == "__main__":
    # The Deep CFR paper's FHP has $100 bets on both rounds (its Appendix A; DREAM's appendix says the same); the class in the
    # repository doubles the bet on the flop.  REF_FLOP_BET=200 keeps the repository's game.
    Flop3Holdem.BIG_BET = int(os.environ.get("REF_FLOP_BET", "100"))
    n_iters, name, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    trav, batches, batch, buf = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(float(sys.argv[7]))
    t_prof = TrainingProfile(
        name=name, n_traversals_per_iter=trav, sampler="es", game_cls=Flop3Holdem, n_batches_adv_training=batches,
        n_learner_actor_workers=1, mini_batch_size_adv=batch, max_buffer_size_adv=buf, DISTRIBUTED=False, log_verbose=False,
        path_data=out, eval_modes_of_algo=(EvalAgentDeepCFR.EVAL_MODE_SINGLE,), export_each_net=True,
        device_training=os.environ.get("REF_DEVICE", "cpu"))
    t_prof.path_strategy_nets = os.path.join(out, "strategy_nets")  # the attribute export_each_net needs (missing in the repository)
    ctrl = Driver(t_prof=t_prof, eval_methods={}, n_iterations=n_iters)
    ctrl.run()
