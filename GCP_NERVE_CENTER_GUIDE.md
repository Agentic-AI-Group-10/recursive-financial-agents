# GCP_NERVE_CENTER_GUIDE.md

## 1. Provisioning the VM (Compute Engine)
The VM (Virtual Machine) will host your database and your telemetry dashboard.

1.  **Open GCP Console:** Go to [console.cloud.google.com](https://console.cloud.google.com/).
2.  **Select/Create Project:** Ensure your new project (with the $300 credits) is selected in the top-left dropdown.
3.  **Navigate to Compute Engine:** Search for "Compute Engine" in the search bar and select **VM Instances**.
4.  **Create Instance:**
    *   **Name:** `nerve-center-vm`
    *   **Region:** Select `us-central1` or any region close to you.
    *   **Machine Configuration:** Select **E2-medium** (2 vCPU, 4GB memory). This is robust enough for your DB and Dashboard and well-covered by your credits.
    *   **Boot Disk:** Click "Change." Select **Ubuntu** as the Operating System and **22.04 LTS** as the Version. Set size to 20GB.
    *   **Firewall:** Check both **"Allow HTTP traffic"** and **"Allow HTTPS traffic"**.
5.  **Finalize:** Click **Create**. It will take about a minute to spin up.

---

## 2. Database Setup (PostgreSQL + pgvector)
Once the VM status shows a green checkmark, click the **SSH** button in the row to open a terminal window.

1.  **Install Postgres:**
    ```bash
    sudo apt update
    sudo apt install postgresql postgresql-contrib -y
    ```
2.  **Install pgvector:**
    ```bash
    # This installs the vector extension for AI memory
    sudo apt install postgresql-14-pgvector -y
    ```
3.  **Configure the Database:**
    ```bash
    # Switch to the postgres user and enter the shell
    sudo -u postgres psql
    ```
    *Inside the Postgres prompt (`postgres=#`), run these SQL commands:*
    ```sql
    CREATE DATABASE agent_memory;
    \c agent_memory;
    CREATE EXTENSION vector;
    CREATE TABLE telemetry (
        id serial PRIMARY KEY, 
        iteration int, 
        profit float, 
        tokens int, 
        joules float, 
        efficiency_score float,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE ltm_entries (
        id serial PRIMARY KEY, 
        embedding vector(768), 
        content text, 
        metadata jsonb
    );
    \q
    ```

---

## 3. Telemetry Hub Setup (Streamlit)
Still in the SSH window, set up a simple web-based dashboard.

1.  **Install Python requirements:**
    ```bash
    sudo apt install python3-pip -y
    pip3 install streamlit pandas psycopg2-binary plotly
    ```
2.  **Create the App:**
    ```bash
    nano app.py
    ```
    *Paste in a simple dashboard script (Antigravity can help you write a more complex one later). Save with `Ctrl+O`, `Enter`, and exit with `Ctrl+X`.*
3.  **Run the Dashboard:**
    ```bash
    # Run on port 80 so it's accessible via your VM's External IP
    sudo python3 -m streamlit run app.py --server.port 80 --server.address 0.0.0.0
    ```

---

## 4. Linking AI Studio & Antigravity

### Linking AI Studio to GCP
1.  Go to [Google AI Studio](https://aistudio.google.com/).
2.  Click the **Settings** (gear icon) in the bottom-left corner.
3.  Click on **Cloud Project**.
4.  Select your GCP Project ID from the list. 
    *   *Note: This ensures your Gemini API usage is tracked under your project and utilizes the associated quota/billing.*

### Linking Antigravity (Local) to GCP VM
1.  **Install GCloud CLI:** Download and install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) on your local machine.
2.  **Authenticate:** Run `gcloud auth login` in your local terminal.
3.  **SSH Tunneling:** To let Antigravity "see" the database as if it's local, run:
    ```bash
    gcloud compute ssh nerve-center-vm -- -L 5432:localhost:5432
    ```
    *Now, tell Antigravity to connect to Postgres at `localhost:5432` with user `postgres`.*

---

## 5. Adding Your Partner (Collaboration)

### GCP & AI Studio Access
1.  In the GCP Console, search for **IAM & Admin**.
2.  Click **Grant Access**.
3.  **New Principals:** Enter your partner's Gmail address.
4.  **Role:** Select **Project > Editor**. 
    *   *Result: They can now SSH into the VM, see the DB, and select the project in their own AI Studio settings.*

### GitHub Access
1.  Go to your repo on GitHub.
2.  Select **Settings** > **Collaborators**.
3.  Click **Add people** and enter your partner’s GitHub username.

---

## 6. Security Note
*   **API Keys:** Never commit your GCP Service Account keys or Gemini API keys to GitHub. 
*   **Environment Variables:** Use a `.env` file locally. Add `.env` to your `.gitignore` file immediately.