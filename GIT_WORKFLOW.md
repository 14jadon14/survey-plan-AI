# 🚀 Git-Ops Workflow Instructions

Use this guide to connect your local project to GitHub and deploy code changes to Colab.

## 1. Initial Setup (Run Once)
Open your terminal in the `SurveyPlan AI` folder and run this block:

```powershell
# Initialize Git
git init

# Add all code files (Safe because .gitignore excludes data)
git add .

# Initial Commit
git commit -m "Initial commit: Modular YOLO training package"

# ⚠️ REPLACE WITH YOUR ACTUAL REPO URL ⚠️
# git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
# git branch -M main
# git push -u origin main
```

## 2. Daily Workflow (Dev -> Deploy)
When you make changes to `src/` locally:

1.  **Commit & Push**:
    ```powershell
    git add .
    git commit -m "Updated model config"
    git push
    ```

2.  **Train on Colab**:
    -   Open `colab_launcher.ipynb` in Google Colab.
    -   Run the cells. It will automatically `git pull` your latest code and start training.

## 3. Data Management
-   **Dataset**: Keep `data.zip` in your Google Drive root (or `SurveyPlan AI` folder in Drive).
-   **Artifacts**: Training results (`runs/`) will be saved back to Google Drive automatically by the launcher.
