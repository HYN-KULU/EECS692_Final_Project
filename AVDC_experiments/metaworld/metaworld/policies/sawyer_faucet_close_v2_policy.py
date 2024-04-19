import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerFaucetCloseV2Policy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_gripper': obs[3],
            'faucet_pos': obs[4:7],
            'unused_info': obs[7:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({'delta_pos': np.arange(3), 'grab_effort': 3})
        to_xyz,branch_id=self._desired_pos(o_d)
        action['delta_pos'] = move(o_d['hand_pos'],
                                   to_xyz=to_xyz,
                                   p=25.)
        action['grab_effort'] = 1.

        return action.array,branch_id

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_faucet = o_d['faucet_pos'] + np.array([+.04, .0, .03])

        if np.linalg.norm(pos_curr[:2] - pos_faucet[:2]) > 0.04:
            branch_id=0
            return pos_faucet + np.array([.0, .0, .1]),branch_id
        elif abs(pos_curr[2] - pos_faucet[2]) > 0.04:
            branch_id=1
            return pos_faucet,branch_id
        else:
            branch_id=2
            return pos_faucet + np.array([-.1, .05, .0]),branch_id
