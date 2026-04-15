#!/bin/sh
# Run every formatter
isort .
black .
taplo fmt .
