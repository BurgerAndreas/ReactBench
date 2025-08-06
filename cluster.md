# How to use the cluster


### Start an interactive session

Killarney
```bash
salloc -A aip-aspuru -t 60:00:00 -D /project/aip-aspuru/aburger/gad-ff --gres=gpu:l40s:1 --mem=128GB
```


### First time connecting to a cluster

```bash
ssh -Y aburger@killarney.alliancecan.ca
```

Add ssh key to github
```bash
ssh-keygen -t ed25519
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

```bash
git config --global user.name "Max Mustermann"
git config --global user.email "max.mustermann@mail.utoronto.ca"
```


### Project setup
Create a .env file in the root directory and set these variables (adjust as needed):
```bash
touch .env
nano .env
```
```bash
# .env
HOMEROOT=${HOME}/gad-ff
# some scratch space where we can write files during training. can be the same as HOMEROOT
PROJECTROOT=${PROJECT}/gad-ff
# the python environment to use (run `which python` to find it)
PYTHONBIN=${HOME}/gad-ff/gadenv/bin/python
WANDB_ENTITY=...
MPLCONFIGDIR=${PROJECTROOT}/.matplotlib
```


Simlink
```bash
rm -rf ${PROJECT}/.cache
rm -rf ${HOME}/.cache
mkdir -p ${PROJECT}/.cache
# the fake folder we will use
ln -s ${PROJECT}/.cache ${HOME}/.cache
# Check the simlink
ls -la ${PROJECT}/.cache

rm -rf ${PROJECT}/.conda
rm -rf ${HOME}/.conda
mkdir -p ${PROJECT}/.conda
# the fake folder we will use
ln -s ${PROJECT}/.conda ${HOME}/.conda
# Check the simlink
ls -la ${PROJECT}/.conda

rm -rf ${PROJECT}/.mamba
rm -rf ${HOME}/.mamba
mkdir -p ${PROJECT}/.mamba
# the fake folder we will use
ln -s ${PROJECT}/.mamba ${HOME}/.mamba
# Check the simlink
ls -la ${PROJECT}/.mamba
```
