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
"""Install source code for the Tapas paper."""
from distutils import spawn
import glob
import os
import subprocess
import sys

from setuptools import find_packages
from setuptools import setup


def find_protoc():
  """Find the Protocol Compiler."""
  if "PROTOC" in os.environ and os.path.exists(os.environ["PROTOC"]):
    return os.environ["PROTOC"]
  elif os.path.exists("../src/protoc"):
    return "../src/protoc"
  elif os.path.exists("../src/protoc.exe"):
    return "../src/protoc.exe"
  elif os.path.exists("../vsprojects/Debug/protoc.exe"):
    return "../vsprojects/Debug/protoc.exe"
  elif os.path.exists("../vsprojects/Release/protoc.exe"):
    return "../vsprojects/Release/protoc.exe"
  else:
    return spawn.find_executable("protoc")


def needs_update(source, target):
  """Returns wheter target file is old or does not exist."""
  if not os.path.exists(target):
    return True
  if not os.path.exists(source):
    return False
  return os.path.getmtime(source) > os.path.getmtime(target)


def fail(message):
  """Write message to stderr and finish."""
  sys.stderr.write(message + "\n")
  sys.exit(-1)


def generate_proto(protoc, source):
  """Invokes the Protocol Compiler to generate a _pb2.py."""

  target = source.replace(".proto", "_pb2.py")

  if needs_update(source, target):
    print(f"Generating {target}...")

    if not os.path.exists(source):
      fail(f"Cannot find required file: {source}")

    if protoc is None:
      fail("protoc is not installed nor found in ../src. Please compile it "
           "or install the binary package.")

    protoc_command = [protoc, "-I.", "--python_out=.", source]
    if subprocess.call(protoc_command) != 0:
      fail(f"Command fail: {' '.join(protoc_command)}")


def prepare():
  """Find all proto files and generate the pb2 ones."""
  proto_file_patterns = ["./tapas/protos/*.proto"]
  protoc = find_protoc()
  for file_pattern in proto_file_patterns:
    for proto_file in glob.glob(file_pattern, recursive=True):
      generate_proto(protoc, proto_file)


def read(fname):
  return open(
      os.path.join(os.path.dirname(__file__), fname), encoding="utf-8").read()


prepare()
setup(
    name="tapas",
    version="0.0.1.dev",
    packages=find_packages(),
    description="Tapas: Table-based Question Answering.",
    long_description=read("README.md"),
    author="Google Inc.",
    url="https://github.com/google-research/tapas",
    license="Apache 2.0",
    install_requires=read("requirements.txt").strip().split("\n"))
