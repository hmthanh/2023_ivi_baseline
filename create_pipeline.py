import joblib as jl
from sklearn.pipeline import Pipeline

from pymo.parsers import *
from pymo.preprocessing import *

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--dataset_path", type=str, default="genea2023_dataset")
    # args = parser.parse_args()
    # data_dir = args.dataset_path

    parser = BVHParser()
    parsed_example = parser.parse("trn_2022_v1_292.bvh")

    mexp_full = Pipeline([
        ('jtsel', JointSelector(["b_root", "b_spine0", "b_spine1", "b_spine2", "b_spine3", "b_neck0", "b_head", "b_r_shoulder",
                                 "b_r_arm", "b_r_arm_twist",
                                 "b_r_forearm", "b_r_wrist_twist",
                                 "b_r_wrist", "b_l_shoulder",
                                 "b_l_arm", "b_l_arm_twist",
                                 "b_l_forearm", "b_l_wrist_twist",
                                 "b_l_wrist", "b_r_upleg", "b_r_leg",
                                 "b_r_foot", "b_l_upleg", "b_l_leg", "b_l_foot"], include_root=True)),
        ('param', MocapParameterizer('expmap')),
        ('cnst', ConstantsRemover_withroot()),
        ('np', Numpyfier()),
    ])
    
    fullexpdata = mexp_full.fit_transform([parsed_example])[0]

    mexp_upperbody = Pipeline([
        ('jtsel', JointSelector(["b_root", "b_spine0", "b_spine1", "b_spine2", "b_spine3", "b_neck0", "b_head", "b_r_shoulder",
                                 "b_r_arm",
                                 "b_r_arm_twist",
                                 "b_r_forearm",
                                 "b_r_wrist_twist",
                                 "b_r_wrist", "b_l_shoulder",
                                 "b_l_arm",
                                 "b_l_arm_twist",
                                 "b_l_forearm",
                                 "b_l_wrist_twist",
                                 "b_l_wrist"
                                 ], include_root=False)),
        ('param', MocapParameterizer('expmap')),
        ('cnst', ConstantsRemover_()),
        ('np', Numpyfier()),
    ])
    upperexpdata = mexp_upperbody.fit_transform([parsed_example])[0]
    
    jl.dump(mexp_full, "./output/data/pipeline_expmap_full.sav")
    jl.dump(mexp_upperbody, "./output/data/pipeline_expmap_upper.sav")
    