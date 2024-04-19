import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerButtonPressTopdownWallV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'hand_closed': obs[3],
            'button_pos': obs[4:7],
            'unused_info': obs[7:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })
        to_xyz,branch_id=self._desired_pos(o_d)
        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=to_xyz, p=25.)
        action['grab_effort'] = -1.

        return action.array,branch_id

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_button = o_d['button_pos'] + np.array([.0, -.06, .0])

        if np.linalg.norm(pos_curr[:2] - pos_button[:2]) > 0.04:
            return pos_button + np.array([0., 0., 0.1]),0
        else:
            return pos_button,1
