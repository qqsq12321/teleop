"""Robot-specific link and URDF configurations for supported hands.

Each entry maps a robot type name to a dict containing:
- origin_link: Base/palm link name
- tip_links: Fingertip link names
- link1_names: Proximal phalanx link names
- link3_names: Middle/PIP link names
- link4_names: Distal/DIP link names
- urdf_subdir, urdf_file: URDF asset paths
- mjcf_subdir, mjcf_file: MuJoCo XML asset paths (optional)
- num_fingers: Number of fingers (4 or 5)
- neutral_qpos: Optional neutral joint position
- Offsets: link3_offsets, tip_offsets, link4_offsets (optional)
"""

ROBOT_CONFIGS = {
    # Shadow Hand (MuJoCo Menagerie style, rh_/lh_ prefix) - high quality meshes
    # Uses custom URDF that exactly matches MuJoCo Menagerie joint axes
    'shadow_hand': {
        'origin_link': 'rh_palm',  # Will be lh_palm for left hand
        'tip_links': ['rh_thtip', 'rh_fftip', 'rh_mftip', 'rh_rftip', 'rh_lftip'],
        'link1_names': ['rh_thproximal', 'rh_ffproximal', 'rh_mfproximal', 'rh_rfproximal', 'rh_lfproximal'],
        'link3_names': ['rh_thmiddle', 'rh_ffmiddle', 'rh_mfmiddle', 'rh_rfmiddle', 'rh_lfmiddle'],
        'link4_names': ['rh_thdistal', 'rh_ffdistal', 'rh_mfdistal', 'rh_rfdistal', 'rh_lfdistal'],
        'urdf_subdir': 'assets/shadow_hand',
        'urdf_file': {'right': 'right_hand_mj.urdf', 'left': 'left_hand_mj.urdf'},
        'mjcf_subdir': 'assets/shadow_hand',
        'mjcf_file': {'right': 'scene_right.xml', 'left': 'scene_left.xml'},
        'num_fingers': 5,
    },
    # Wuji Hand (5 fingers x 4 joints = 20 DOF)
    'wuji_hand': {
        'origin_link': 'right_palm_link',
        'tip_links': ['right_finger1_tip_link', 'right_finger2_tip_link', 'right_finger3_tip_link', 'right_finger4_tip_link', 'right_finger5_tip_link'],
        'link1_names': ['right_finger1_link1', 'right_finger2_link1', 'right_finger3_link1', 'right_finger4_link1', 'right_finger5_link1'],
        'link3_names': ['right_finger1_link3', 'right_finger2_link3', 'right_finger3_link3', 'right_finger4_link3', 'right_finger5_link3'],
        'link4_names': ['right_finger1_link4', 'right_finger2_link4', 'right_finger3_link4', 'right_finger4_link4', 'right_finger5_link4'],
        'num_fingers': 5,
    },
    # Allegro Hand (4 fingers: thumb, index, middle, ring - no pinky)
    # Finger order: thumb (link_12~15), index (link_0~3), middle (link_4~7), ring (link_8~11)
    'allegro_hand': {
        'origin_link': 'base_link',
        'tip_links': ['link_15.0_tip', 'link_3.0_tip', 'link_7.0_tip', 'link_11.0_tip'],
        'link1_names': ['link_12.0', 'link_0.0', 'link_4.0', 'link_8.0'],
        'link3_names': ['link_13.0', 'link_1.0', 'link_5.0', 'link_9.0'],
        'link4_names': ['link_14.0', 'link_2.0', 'link_6.0', 'link_10.0'],
        'num_fingers': 4,
    },
    # Inspire Hand (5 fingers, 2-DOF per non-thumb finger)
    # Non-thumb: proximal -> intermediate -> tip(fixed), so link3=proximal(PIP), link4=intermediate(DIP)
    'inspire_hand': {
        'origin_link': 'hand_base_link',
        'tip_links': ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip'],
        'link1_names': ['thumb_proximal', 'index_proximal', 'middle_proximal', 'ring_proximal', 'pinky_proximal'],
        'link3_names': ['thumb_proximal', 'index_proximal', 'middle_proximal', 'ring_proximal', 'pinky_proximal'],
        'link4_names': ['thumb_intermediate', 'index_intermediate', 'middle_intermediate', 'ring_intermediate', 'pinky_intermediate'],
        'num_fingers': 5,
    },
    # Ability Hand (5 fingers, 2 links each)
    'ability_hand': {
        'origin_link': 'base',
        'tip_links': ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip'],
        'link1_names': ['thumb_L1', 'index_L1', 'middle_L1', 'ring_L1', 'pinky_L1'],
        'link3_names': ['thumb_L1', 'index_L1', 'middle_L1', 'ring_L1', 'pinky_L1'],
        'link4_names': ['thumb_L2', 'index_L2', 'middle_L2', 'ring_L2', 'pinky_L2'],
        'num_fingers': 5,
    },
    # Leap Hand (4 fingers + thumb, no pinky)
    'leap_hand': {
        'origin_link': 'base',
        'tip_links': ['thumb_tip_head', 'index_tip_head', 'middle_tip_head', 'ring_tip_head'],
        'link1_names': ['thumb_pip', 'pip', 'pip_2', 'pip_3'],
        'link3_names': ['thumb_dip', 'dip', 'dip_2', 'dip_3'],
        'link4_names': ['thumb_fingertip', 'fingertip', 'fingertip_2', 'fingertip_3'],
        'num_fingers': 4,
    },
    # SVH Hand (5 fingers)
    'svh_hand': {
        'origin_link': 'right_hand_base_link',
        'tip_links': ['thtip', 'fftip', 'mftip', 'rftip', 'lftip'],
        'link1_names': ['right_hand_z', 'right_hand_l', 'right_hand_k', 'right_hand_j', 'right_hand_i'],
        'link3_names': ['right_hand_a', 'right_hand_p', 'right_hand_o', 'right_hand_n', 'right_hand_m'],
        'link4_names': ['right_hand_b', 'right_hand_t', 'right_hand_s', 'right_hand_r', 'right_hand_q'],
        'num_fingers': 5,
    },
    # LinkerHand L21
    'linkerhand_l21': {
        'origin_link': 'hand_base_link',
        'tip_links': ['thumb_distal', 'index_middle', 'middle_middle', 'ring_middle', 'pinky_middle'],
        'link1_names': ['thumb_metacarpals', 'index_metacarpals', 'middle_metacarpals', 'ring_metacarpals', 'pinky_metacarpals'],
        'link3_names': ['thumb_metacarpals', 'index_proximal', 'middle_proximal', 'ring_proximal', 'pinky_proximal'],
        'link4_names': ['thumb_distal', 'index_middle', 'middle_middle', 'ring_middle', 'pinky_middle'],
        'link3_offsets': [
            [0.018, 0.000, 0.000],
            [0.000, 0.000, 0.022],
            [0.000, 0.000, 0.022],
            [0.000, 0.000, 0.022],
            [0.000, 0.000, 0.022],
        ],
        'tip_offsets': [
            [0.040, 0.000, 0.000],
            [0.000, 0.000, 0.044],
            [0.000, 0.000, 0.044],
            [0.000, 0.000, 0.044],
            [0.000, 0.000, 0.044],
        ],
        'link4_offsets': [
            [0.023, 0.000, 0.000],
            [0.000, 0.000, 0.022],
            [0.000, 0.000, 0.022],
            [0.000, 0.000, 0.022],
            [0.000, 0.000, 0.022],
        ],
        'urdf_subdir': 'assets/linkerhand_l21',
        'urdf_file': {
            'right': 'right/linkerhand_l21_right_vis.urdf',
            'left': 'left/linkerhand_l21_left_vis.urdf',
        },
        'num_fingers': 5,
        'neutral_qpos': [0.0] * 17,
    },
    # ROHand
    'rohand': {
        'origin_link': 'base_link',
        'tip_links': ['th_distal_link', 'if_distal_link', 'mf_distal_link', 'rf_distal_link', 'lf_distal_link'],
        'link1_names': ['th_root_link', 'if_slider_abpart_link', 'mf_slider_abpart_link', 'rf_slider_abpart_link', 'lf_slider_abpart_link'],
        'link3_names': ['th_root_link', 'if_slider_abpart_link', 'mf_slider_abpart_link', 'rf_slider_abpart_link', 'lf_slider_abpart_link'],
        'link4_names': ['th_proximal_link', 'if_proximal_link', 'mf_proximal_link', 'rf_proximal_link', 'lf_proximal_link'],
        'urdf_subdir': 'assets/rohand',
        'urdf_file': {
            'right': 'right/rohand_right_vis.urdf',
            'left': 'left/rohand_left_vis.urdf',
        },
        'num_fingers': 5,
    },
    # Unitree Dex5
    'unitree_dex5_hand': {
        'origin_link': 'base_link00',
        'tip_links': ['Link_14R', 'Link_24R', 'Link_34R', 'Link_44R', 'Link_54R'],
        'link1_names': ['Link_11R', 'Link_21R', 'Link_31R', 'Link_41R', 'Link_51R'],
        'link3_names': ['Link_12R', 'Link_22R', 'Link_32R', 'Link_42R', 'Link_52R'],
        'link4_names': ['Link_13R', 'Link_23R', 'Link_33R', 'Link_43R', 'Link_53R'],
        'urdf_subdir': 'assets/unitree_dex5_hand',
        'urdf_file': {
            'right': 'right/Dex5-URDF-R.urdf',
            'left': 'left/Dex5-URDF-L.urdf',
        },
        'num_fingers': 5,
        'neutral_qpos': [0.0] * 20,
    },
}
