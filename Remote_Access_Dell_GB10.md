# Remote access to Dell GB10 (quick guide)

This document describes supported ways to remotely access a Dell GB10 system.  
The **primary and recommended workflow** is SSH with headless Jupyter Lab.  
An **AnyDesk remote desktop option** is documented as a fallback.



## Prerequisites

- Account on the GB10 host  
- UW Network access (use GlobalProtect VPN from UW-Madison if working off-campus)
- Local SSH client  
- Assigned GPU access (typically single-user per box)



## Primary workflow (recommended): SSH + headless Jupyter

### 1. SSH into the GB10

```bash
# ssh <username>@<gb10-hostname>
ssh mlx@128.104.18.206 # ethernet address
```

If the above fails or times out, try connecting to the Wi-Fi address instead.

```bash
ssh mlx@10.141.72.249 # wifi
```

Notes:
- If prompted with "The authenticity of host ...", type "yes" to continue
- You may need to use VPN if accessing from outside the local network
- Once logged in, you are at a command line on the GB10
- Treat the system as a headless Linux workstation



### 2. Project setup and virtual environment (template)

This mirrors the GB10 test workflow used in earlier evaluations.

Clone the repository:

```bash
cd GitHub/ # where we store git projects on this machine
git clone https://github.com/qualiaMachine/GB10_Tests.git
```

Change into the project directory:

```bash
cd GB10_Tests/WattBot
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

Create a virtual environment (creates `.venv`):

```bash
uv venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

Install requirements:

```bash
uv pip install -r requirements.txt
```

Register the venv as a named Jupyter kernel:

```bash
python -m ipykernel install   --user   --name wattbot   --display-name "wattbot"
```



### 3. Launch Jupyter Lab (headless)

```bash
jupyter lab --no-browser --port=8888
```

Keep this SSH session open.


### 4. Forward the Jupyter port to your local machine

From a second **local** terminal:

```bash
ssh -N -L 8888:localhost:8888 mlx@10.141.72.249
```

Then open in your local browser:

```
http://localhost:8888
```

You will be prompted for Jupyter authentication.

By default, Jupyter Lab starts with token-based authentication.
In the SSH session where you launched Jupyter (1st terminal), look for a line like:

```
http://127.0.0.1:8888/lab?token=xxxxxxxxxxxxxxxx
```

Copy just the token=...
value and paste it into the login field.

If you no longer have the token visible, run this on the GB10:

## Backup workflow: AnyDesk remote desktop (optional)

AnyDesk may be used as a fallback if SSH port forwarding is not viable or if a lightweight GUI is helpful.

### Requirements

- AnyDesk installed locally: https://anydesk.com/en
- AnyDesk installed and enabled on the GB10  
- Access approval or credentials

### Steps

1. Launch AnyDesk on your local machine.
2. Enter the GB10's AnyDesk address (ask Chris if unknown).
3. Connect and provide approval or credentials as needed.
4. Once connected, you will see the GB10 desktop environment.
