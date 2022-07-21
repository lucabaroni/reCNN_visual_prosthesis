import wandb
from model_trainer import (
    Lurz_dataset_preparation_function,
    Antolik_dataset_preparation_function_test,
)
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import argparse

from create_ensemble import download_model, create_ensemble, download_control_model
from num_of_parameters import return_number_of_parameters

def present_results(args, lists_of_model_names, control_model_name, data_module, trainer, run, model_name_prefix="csng-cuni/reCNN_visual_prosthesis/model-"):
    """
        lists_of_model_names:
            - list of lists of best models for a particular task. The first
              element in the sublist is a string with a short description of
              the task. En example of the list:
              [["The best model with 16 rotations trained on dataset from Lurz et al.", "model/name/in/wandb"],
               ["Ensemble of 2 models with 16 rotations trained on dataset from Lurz et al.", "model1/name/in/wandb", "model1/name/in/wandb"]
    """

    config = {
        "lr": 0.001,
        "test_average_batch": False,
        "compute_oracle_fraction": False,
        "conservative_oracle": True,
        "jackknife_oracle": True,
        "generate_oracle_figure": False,
        "batch_size": 10,
    }

    control_measures = None

    if control_model_name != None:
        print("------------------------------------------------------------------------")
        print()
        print("CONTROL MODEL:")
        print()

        control_model = download_control_model(model_name_prefix + control_model_name + ":v0", run)
        print(f"Number of parameters in the core is {return_number_of_parameters(control_model, 'core')}.")
        control_model.freeze()
        control_measures = data_module.model_performances(control_model, trainer)
        
        print()
        print("------------------------------------------------------------------------")

    for task in lists_of_model_names:
        print("------------------------------------------------------------------------")
        print()

        task_name = task[0]

        print(task_name)
        print()

        model = None
        if len(task) == 2:
            model = download_model(model_name_prefix + task[1] + ":v0", run)
            print(f"Number of parameters in the core is {return_number_of_parameters(model, 'core')}.")
            model.freeze()
        else:
            model = create_ensemble(task[1:], config, run, model_name_prefix, ":v0")
        
        
        data_module.model_performances(model, trainer, control_measures)

        print()
        print("------------------------------------------------------------------------")


def present_lurz(args, trainer, run):

    print("Presenting models trained on the dataset from Lurz et al.")
    dm = Lurz_dataset_preparation_function({"batch_size": 10})

    control_model_name = "8qqi18gl"

    list_of_model_names = [
        ["The best model with 8 rotations trained on dataset from Lurz et al.", "unpwsnba"],
        ["Ensemble of 5 models with 8 rotations trained on dataset from Lurz et al.", "v9gfoto8", "q7zrzwfs", "uwl9k6yy", "goe2mitq", "unpwsnba"],
        ["The best model with 16 rotations trained on dataset from Lurz et al.", "zipjy41m"],
        ["Ensemble of 5 models with 16 rotations trained on dataset from Lurz et al.", "dp7c43r5", "wkw2tizy", "iq8yo22o", "wh88cya8", "zipjy41m"],
        ["The best model with 24 rotations trained on dataset from Lurz et al.", "hl2gbqtm"],
        ["Ensemble of 5 models (not necessarily with the same architecture) with 24 rotations trained on dataset from Lurz et al.", "erbca5lq", "fnxp9gzu", "sxdmolgo", "6859220j", "hl2gbqtm"],
    ]
    
    present_results(args, list_of_model_names, control_model_name, dm, trainer, run)


def present_antolik(args, trainer, run):
    print("Presenting models trained on the synthetic dataset generated by a model from Antolík et al.")

    dm = Antolik_dataset_preparation_function_test({"batch_size": 10})

    control_model_name = "3szw28mw"

    list_of_model_names = [
        ["The best model with 8 rotations trained on dataset from Lurz et al.", "abwwf3vs"],
        ["Ensemble of 5 models with 8 rotations trained on dataset from Lurz et al.", "ir6d5s39", "hlfxqbpz", "bzt64a1d", "uw8rl69f", "abwwf3vs"],
        ["The best model with 16 rotations trained on dataset from Lurz et al.", "1f5x6dz0"],
        ["Ensemble of 5 models with 16 rotations trained on dataset from Lurz et al.", "h4etwje7", "pvm7l1fg", "cahvhpyr", "kmhokhhq", "1f5x6dz0"],
    ]
    
    present_results(args, list_of_model_names, control_model_name, dm, trainer, run)
    

def main(args):

    wandb_logger = WandbLogger(log_model=True)
    trainer = pl.Trainer(gpus=[0], logger=wandb_logger)
    run = wandb.init(project="reCNN_visual_prosthesis", entity="csng-cuni")
    
    if args.dataset_type == "both":
        present_lurz(args, trainer, run)
        present_antolik(args, trainer, run)
    elif args.dataset_type in ["lurz", "Lurz"]:
        present_lurz(args, trainer, run)
    elif args.dataset_type in ["antolik", "Antolik"]:
        present_antolik(args, trainer, run)
    else:
        raise Exception("Wrong dataset type.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_type",
        default="Lurz",
        type=str,
        help="The presented dataset. Possible options are: Lurz, Antolik, both",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=True,
        help="Whether to use Weights & Biases or not.",
    )

    args = parser.parse_args()
    
    main(args)