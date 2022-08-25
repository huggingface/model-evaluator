import os
from pathlib import Path

import typer
from datasets import load_dataset
from dotenv import load_dotenv
from rich import print

from utils import http_get, http_post

if Path(".env").is_file():
    load_dotenv(".env")

HF_TOKEN = os.getenv("HF_TOKEN")
AUTOTRAIN_TOKEN = os.getenv("AUTOTRAIN_TOKEN")
AUTOTRAIN_USERNAME = os.getenv("AUTOTRAIN_USERNAME")
AUTOTRAIN_BACKEND_API = os.getenv("AUTOTRAIN_BACKEND_API")


def main():
    logs_df = load_dataset("autoevaluate/evaluation-job-logs", use_auth_token=True, split="train").to_pandas()
    evaluated_projects_ds = load_dataset("autoevaluate/evaluated-project-ids", use_auth_token=True, split="train")
    projects_df = logs_df.copy()[(~logs_df["project_id"].isnull()) & (logs_df["is_evaluated"] == False)]
    projects_to_approve = projects_df["project_id"].astype(int).tolist()

    for project_id in projects_to_approve:
        project_status = http_get(
            path=f"/projects/{project_id}",
            token=HF_TOKEN,
            domain=AUTOTRAIN_BACKEND_API,
        ).json()
        if project_status["status"] == 3:
            train_job_resp = http_post(
                path=f"/projects/{project_id}/start_training",
                token=HF_TOKEN,
                domain=AUTOTRAIN_BACKEND_API,
            ).json()
            print(f"🏃‍♂️ Project {project_id} approval response: {train_job_resp}")
            # if train_job_resp["approved"] == True:
            #     # Update evaluation status


if __name__ == "__main__":
    typer.run(main)
