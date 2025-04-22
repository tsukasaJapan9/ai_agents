#!/bin/bash

sudo ps aux | grep "llm" | grep -v grep | awk '{print $2}' | xargs kill -9
sudo ps aux | grep "ui" | grep -v grep | awk '{print $2}' | xargs kill -9
