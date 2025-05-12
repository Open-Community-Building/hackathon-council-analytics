import os
import time
from dagster import AssetExecutionContext, AssetSelection, AutomationCondition, DagsterEventType, Definitions, EventRecordsFilter, RunRequest, SensorEvaluationContext, SensorResult, SkipReason, asset, DynamicPartitionsDefinition, sensor

import pymupdf
import requests
from typing import Optional


def get_content_type(url):
    response = requests.get(url, stream=True)
    if response.status_code == 404:
        return None
    
    return response.headers.get('content-type')

def check_pdf(url, idx):
    url = f"{url}?id={idx}&type=do"

    content_type = get_content_type(url)

    if content_type is not None and 'application/pdf' in content_type:
        return True
    else:
        return False


def request_pdf(url, idx) -> str:
    """
    Request the PDF file from the municipal council website.
    Returns the content of the file if it's a PDF, otherwise None.
    """
    url = f"{url}?id={idx}&type=do"
    response = requests.get(url, stream=True)
    if response.status_code == 404:
        print(f"no file for {idx}")
        return None
    content_type = response.headers.get('content-type')

    if 'application/pdf' in content_type:
        print(f"PDF found for {idx}.")
        return response.content
    else:
        print(f"The file retrieved for id {idx} is not a PDF.")
        return None


def extract_text(doc):
    text = ""
    for page in doc:
        text += page.get_text()
    return text

chunk_partitions = DynamicPartitionsDefinition(name="chunks")


@asset(partitions_def=chunk_partitions)
def pdfs(context: AssetExecutionContext):
    url = 'https://www.gemeinderat.heidelberg.de/getfile.asp'

    context.log.info(f"context info: {context.partition_key}")
    start, end = [int(i) for i in context.partition_key.split('-')]
    for idx in range(start, end):
        pdf_content = request_pdf(url, idx)
        if pdf_content:
            print(f"PDF {idx} downloaded from source.")
            filename = f"./data/{idx}.pdf"
            with open(filename, 'wb') as f:
                f.write(pdf_content)


@asset(deps=[pdfs], partitions_def=chunk_partitions, automation_condition=AutomationCondition.on_missing())
# @asset(deps=[pdfs], partitions_def=chunk_partitions)
def extracted_text(context: AssetExecutionContext):
    start, end = [int(i) for i in context.partition_key.split('-')]
    for idx in range(start, end):
        pdf_filename = f"./data/{idx}.pdf"
        if os.path.exists(pdf_filename):
            doc = pymupdf.open(pdf_filename, filetype="pdf")
            text = extract_text(doc)
            text_filename = f"./extracted_text/{idx}.txt"
            with open(text_filename, 'w') as f:
                f.write(text)
            context.log.info(f"Extracted text from {pdf_filename}")

# @asset
# def llm_asset


@sensor(asset_selection=AssetSelection.keys(pdfs.key), minimum_interval_seconds=60)
def pdf_sensor(context: SensorEvaluationContext):
    # context.instance
    # get last partition id otherwise its = 0
    # maximum = last_id + 1000, check backward.
    records = context.instance.get_event_records(
        event_records_filter=EventRecordsFilter(
            asset_key=pdfs.key,
            event_type=DagsterEventType.ASSET_MATERIALIZATION
        ),
        limit=1,
        ascending=False
    )

    new_partition_start = 367800
    if records:
        latest_event = records[0]
        new_partition_start = int(latest_event.partition_key.split('-')[1]) + 1
        context.log.info(f"partition key: {latest_event.partition_key}")

    # Check 1000 partitions ahead:
    url = 'https://www.gemeinderat.heidelberg.de/getfile.asp'
    rng = 100
    max_partition_end = new_partition_start + rng
    for idx in range(max_partition_end - 1, new_partition_start - 1, -1):
        if check_pdf(url, idx):
            new_partition_end = idx
            new_partition = f'{new_partition_start}-{new_partition_end}'
            return SensorResult(
                run_requests=[
                    RunRequest(partition_key=new_partition, run_key=f"{pdfs.key}-{new_partition}")
                ],
                dynamic_partitions_requests=[
                    chunk_partitions.build_add_request([new_partition])
                ]
            )
        
    return SkipReason("No new pdf")



# defs = Definitions(assets=[pdfs], sensors=[pdf_sensor])
defs = Definitions(assets=[pdfs, extracted_text], sensors=[pdf_sensor])



# Write a schedule, to trigger daily
# How do you check for the lastest file id in that day? Maximum = last_id + 1000, check backward.
# Naming for the partition: start_id-end_id
# How can the schedule get the current id?
#### Maybe read it from the asset metadata? 

# What is asset now? Asset is a group of heidelberg document, with dynamic partitions:
    # The partition formular: start_id-end_id, start from 1 to 1000, 1000 to 2000
    # then the new partition will be added everytime the scheduler detect new file.
