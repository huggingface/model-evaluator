import os
from pathlib import Path

import typer
from datasets import load_dataset
from dotenv import load_dotenv

from utils import http_get, http_post

if Path(".env").is_file():
    load_dotenv(".env")

HF_TOKEN = os.getenv("HF_TOKEN")
AUTOTRAIN_USERNAME = os.getenv("AUTOTRAIN_USERNAME")
AUTOTRAIN_BACKEND_API = os.getenv("AUTOTRAIN_BACKEND_API")

if "staging" in AUTOTRAIN_BACKEND_API:
    AUTOTRAIN_ENV = "staging"
else:
    AUTOTRAIN_ENV = "prod"


def main():
    print(f"💡 Starting jobs on {AUTOTRAIN_ENV} environment")
    logs_df = load_dataset("autoevaluate/evaluation-job-logs", use_auth_token=HF_TOKEN, split="train").to_pandas()
    # Filter out legacy AutoTrain submissions prior to project approvals requirement
    projects_df = logs_df.copy()[(~logs_df["project_id"].isnull())]
    # Filter IDs for appropriate AutoTrain env (staging vs prod)
    projects_df = projects_df.copy().query(f"autotrain_env == '{AUTOTRAIN_ENV}'")
    projects_to_approve = projects_df["project_id"].astype(int).tolist()
    failed_approvals = []
    print(f"🚀 Found {len(projects_to_approve)} evaluation projects to approve!")

    for project_id in projects_to_approve:
        print(f"Attempting to evaluate project ID {project_id} ...")
        try:
            project_info = http_get(
                path=f"/projects/{project_id}",
                token=HF_TOKEN,
                domain=AUTOTRAIN_BACKEND_API,
            ).json()
            print(project_info)
            # Only start evaluation for projects with completed data processing (status=3)
            if project_info["status"] == 3 and project_info["training_status"] == "not_started":
                train_job_resp = http_post(
                    path=f"/projects/{project_id}/start_training",
                    token=HF_TOKEN,
                    domain=AUTOTRAIN_BACKEND_API,
                ).json()
                print(f"🤖 Project {project_id} approval response: {train_job_resp}")
            else:
                print(f"💪 Project {project_id} either not ready or has already been evaluated. Skipping ...")
        except Exception as e:
            print(f"There was a problem obtaining the project info for project ID {project_id}")
            print(f"Error message: {e}")
            failed_approvals.append(project_id)
            pass

    if len(failed_approvals) > 0:
        print(f"🚨 Failed to approve {len(failed_approvals)} projects: {failed_approvals}")


if __name__ == "__main__":
    typer.run(main)
