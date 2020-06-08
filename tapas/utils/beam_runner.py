# coding=utf-8
# Copyright 2019 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""Utilities for running beam pipelines."""

import enum

from absl import flags
from apache_beam import runners
from apache_beam.options import pipeline_options
from apache_beam.runners.direct import direct_runner



class RunnerType(enum.Enum):
  DIRECT = 1
  DATAFLOW = 2


flags.DEFINE_enum_class("runner_type", RunnerType.DIRECT, RunnerType,
                        "Runner type to use.")
# Google Cloud options.
# See https://beam.apache.org/get-started/wordcount-example/
flags.DEFINE_string("gc_project", None, "e.g. my-project-id")
# GC regions: https://cloud.google.com/compute/docs/regions-zones
flags.DEFINE_string("gc_region", None, "e.g. us-central1")
flags.DEFINE_string("gc_job_name", None, "e.g. myjob")
flags.DEFINE_string("gc_staging_location", None,
                    "e.g. gs://your-bucket/staging")
flags.DEFINE_string("gc_temp_location", None, "e.g. gs://your-bucket/temp")
# Pass Tapas sources to GC.
# See https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/
flags.DEFINE_list(
    "extra_packages",
    None,
    "Packed Tapas sources (python3 setup.py sdist).",
)

FLAGS = flags.FLAGS


def run_type(pipeline, runner_type):
  """Executes pipeline with certain runner type."""
  if runner_type == RunnerType.DIRECT:
    print("Running pipeline with direct runner this might take a long time!")
    return direct_runner.DirectRunner().run(pipeline)
  if runner_type == RunnerType.DATAFLOW:
    options = pipeline_options.PipelineOptions()
    gc_options = options.view_as(pipeline_options.GoogleCloudOptions)
    gc_options.project = FLAGS.gc_project
    gc_options.region = FLAGS.gc_region
    gc_options.job_name = FLAGS.gc_job_name
    gc_options.staging_location = FLAGS.gc_staging_location
    gc_options.temp_location = FLAGS.gc_temp_location
    setup = options.view_as(pipeline_options.SetupOptions)
    setup.extra_packages = FLAGS.extra_packages
    return runners.DataflowRunner().run(pipeline, options=options)
  raise ValueError(f"Unsupported runner type: {runner_type}")


def run(pipeline):
  return run_type(pipeline, FLAGS.runner_type)
