FROM continuumio/miniconda3:latest

COPY environment.yml /environment.yml

RUN conda env create -f /environment.yml