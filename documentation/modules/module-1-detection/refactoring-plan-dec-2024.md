# Refactoring Plan: Module 1 Documentation & Code Cleanup
**Project:** Container ID Extraction Research  
**Module:** Module 1 - Container Door Detection  
**Version:** 1.0  
**Created:** 2024-12-09  
**Purpose:** Align documentation and code with current GitHub clone workflow (deprecate SSH tunnel references)

---

## Executive Summary

### Current State Assessment

**Workflow Evolution:**
- **Old Method (Dec 2024):** SSH tunnel via cloudflared/ngrok → Remote development on Kaggle VM
- **New Method (Current):** Direct notebook workflow → GitHub clone → pyproject.toml install → Training
- **Migration Status:** Code implementation complete, documentation partially outdated

**Deprecation Trigger:**
- SSH tunnel method deprecated Dec 9, 2024
- Root cause: GPU driver incompatibility (Error 803), Poetry environment overhead, maintenance complexity
- Archived in: `documentation/archive/deprecated-ssh-method/`

**Current Issues:**
- **50+ SSH/tunnel references** scattered across documentation files
- Core technical specifications describe deprecated architecture
- Implementation plan contains outdated workflow instructions
- Shell scripts marked deprecated but still referenced in docs

### Scope of Changes

| Category                          | Files Affected                  | Estimated Effort |
| --------------------------------- | ------------------------------- | ---------------- |
| **Critical Documentation**        | 2 files (1 major rewrite)       | 8-12 hours       |
| **High Priority Documentation**   | 3 files (section updates)       | 4-6 hours        |
| **Medium Priority Documentation** | 2 files (minor updates)         | 1-2 hours        |
| **Code Cleanup**                  | 2 scripts (deprecation marking) | 1 hour           |
| **Validation**                    | All files                       | 2-3 hours        |
| **Total**                         | 9 files                         | **16-24 hours**  |

---

## Phase 1: Critical Documentation (Priority: URGENT)

### 1.1 Rewrite `technical-specification-training.md`

**Status:** SEVERELY OUTDATED - Entire architecture describes SSH tunnel workflow

**Issues Identified:**
- Lines 1-100: Architecture section shows SSH tunnel as primary workflow
- Design Principles: Lists "SSH Remote Development" (no longer applicable)
- System diagram: Shows "Phase 0: SSH Tunnel Setup (Notebook)" with cloudflared
- Execution workflow: Describes SSH terminal commands

**Required Changes:**

#### Section: Architecture Overview (Lines ~20-80)
**Action:** REWRITE architecture to describe Direct Notebook workflow

**OLD (Deprecated):**
```markdown
### Phase 0: SSH Tunnel Setup (Notebook)
1. Run tunnel notebook cells
2. Establish cloudflared/ngrok tunnel
3. Connect via SSH to Kaggle VM
4. Environment variables injected to shell
```

**NEW (Current):**
```markdown
### Phase 0: Repository Setup (Direct Notebook)
1. Clone repository from GitHub (single cell)
2. Install dependencies from pyproject.toml
3. Configure secrets from Kaggle Secrets API
4. Verify GPU availability
```

#### Section: Design Principles (Lines ~90-120)
**Action:** REMOVE "SSH Remote Development" principle, ADD "Single-Cell Notebook Execution"

**Changes:**
- ❌ Remove: "SSH Remote Development enables full-featured IDEs"
- ✅ Add: "Single-Cell Execution: All setup/training in one notebook cell for simplicity"
- ✅ Add: "Native Kaggle Secrets: Use Kaggle Secrets API (no environment variable injection)"

#### Section: System Diagram (Lines ~150-200)
**Action:** REPLACE SSH tunnel diagram with Direct Notebook flowchart

**NEW Diagram:**
```
┌────────────────────────────────────────────────────────┐
│  Kaggle Notebook (GPU Kernel)                          │
├────────────────────────────────────────────────────────┤
│                                                         │
│  Step 1: Clone GitHub Repository                       │
│    git clone https://github.com/user/container-id-research.git │
│                                                         │
│  Step 2: Install Dependencies                          │
│    pip install -e . (from pyproject.toml)              │
│                                                         │
│  Step 3: Configure DVC (Kaggle Secrets)                │
│    DVC_SERVICE_ACCOUNT_JSON → /tmp/dvc_creds.json      │
│                                                         │
│  Step 4: Authenticate WandB                            │
│    WANDB_API_KEY → wandb login                         │
│                                                         │
│  Step 5: Fetch Dataset                                 │
│    dvc pull data/processed/detection.dvc               │
│                                                         │
│  Step 6-8: Training Execution                          │
│    python src/detection/train.py --config params.yaml  │
│                                                         │
│  Step 9: Post-Training Sync (Optional)                 │
│    - dvc add weights/detection/best.pt                 │
│    - git commit metadata                               │
│    - Manual download from Kaggle output                │
│                                                         │
└────────────────────────────────────────────────────────┘
```

#### Section: Execution Instructions (Lines ~500+)
**Action:** REPLACE all SSH terminal commands with notebook cell instructions

**OLD References to Remove:**
- "Via SSH terminal connected to Kaggle VM"
- "Ensure tunnel notebook is running"
- `bash scripts/setup_kaggle.sh` (deprecated script)
- `bash scripts/run_training.sh` (deprecated script)

**NEW Instructions:**
```markdown
### Execution Workflow (Direct Notebook)

**File:** `kaggle_training_notebook.py`

**Usage:**
1. Upload `kaggle_training_notebook.py` to Kaggle
2. Create new notebook, enable GPU (T4 or P100)
3. Copy entire file content into single cell
4. Configure Kaggle Secrets (see KAGGLE_TRAINING_GUIDE.md → Section: 🔑 DVC Session Token Setup)
5. Run cell (estimated time: 3-4 hours for 150 epochs)

**Output Location:** `weights/detection/weights/best.pt`
```

**Estimated Effort:** 6-8 hours (major rewrite with diagram creation)

**Dependencies:** None

**Validation:**
- [ ] No SSH/tunnel references remain
- [ ] Architecture diagram reflects current workflow
- [ ] All code examples use notebook cell format (not bash commands)
- [ ] Links to `kaggle_training_notebook.py` correct

---

### 1.2 Update `implementation-plan.md`

**Status:** CONTAINS 40+ SSH REFERENCES - Outdated task descriptions

**Issues Identified:**
- Lines 50-200: Task 1.1, 1.2 describe SSH tunnel setup and scripts
- Multiple references to "SSH session", "SSH terminal"
- Task descriptions assume SSH workflow (setup_kaggle.sh, run_training.sh)
- Execution context instructions reference tunnel notebook

**Required Changes:**

#### Section: Phase 1 - Infrastructure Setup (Lines 50-300)
**Action:** REWRITE Task 1.1 and 1.2 to describe Direct Notebook workflow

**Task 1.1 (Current):** "Create `scripts/setup_kaggle.sh`"
**Task 1.1 (NEW):** "Create Direct Notebook Training Cell"

**OLD Description (Lines 50-180):**
```markdown
**Objective:** Automate Kaggle environment setup (dependencies, DVC, WandB) in SSH session
**Dependencies:** SSH tunnel established (Task 1.0 prerequisite)
**Execution Context:** This script runs **inside the SSH session** on Kaggle VM
```

**NEW Description:**
```markdown
**Objective:** Implement single-cell notebook for complete training workflow (GitHub clone → train → sync)
**Dependencies:** Kaggle Secrets configured (DVC, GitHub Token, WandB)
**Execution Context:** Single notebook cell executed in Kaggle GPU kernel
**Reference:** `kaggle_training_notebook.py`
```

**Task 1.2 (Current):** "Create `scripts/run_training.sh`"
**Action:** MARK AS DEPRECATED, replace with "Manual Model Download Workflow"

**NEW Task 1.2:** "Configure Post-Training Model Download"
```markdown
**Objective:** Document workflow for downloading trained models from Kaggle to local machine

**Steps:**
1. After training completes, download output from Kaggle:
   - Navigate to Notebook → Output → Download All
   - Extract `weights/detection/weights/best.pt`
2. Local DVC push:
   ```bash
   dvc add weights/detection/best.pt
   git add weights/detection/best.pt.dvc .gitignore
   git commit -m "feat(detection): add trained model vX.X"
   dvc push
   git push
   ```

**Limitation:** DVC push from Kaggle fails (Service Account cannot write to personal Google Drive)
**Workaround:** Manual download accepted for personal projects
```

#### Section: All Code Examples (Throughout File)
**Action:** REPLACE bash script commands with Python notebook code

**Search-and-Replace Patterns:**

| OLD Pattern                    | NEW Pattern                                                               |
| ------------------------------ | ------------------------------------------------------------------------- |
| `bash scripts/setup_kaggle.sh` | `# Notebook Cell: GitHub Clone + Setup (see kaggle_training_notebook.py)` |
| `bash scripts/run_training.sh` | `# Notebook Cell: Training Execution (see kaggle_training_notebook.py)`   |
| `Via SSH terminal`             | `In Kaggle notebook cell`                                                 |
| `SSH session`                  | `Kaggle GPU kernel`                                                       |
| `tunnel notebook`              | `training notebook`                                                       |
| `setup_kaggle.sh`              | `kaggle_training_notebook.py (Step 1-5)`                                  |
| `run_training.sh`              | `kaggle_training_notebook.py (Step 6-9)`                                  |

#### Section: Verification Criteria (Throughout File)
**Action:** UPDATE test commands to reflect notebook execution

**OLD:**
```markdown
1. SSH connected to Kaggle VM successfully
2. Environment variables visible: `echo $KAGGLE_SECRET_DVC_JSON`
3. Script runs without errors
```

**NEW:**
```markdown
1. Kaggle notebook cell executes without errors
2. GPU detected: "Device: cuda (NVIDIA Tesla T4)"
3. Dataset fetched successfully (831 images)
4. Training starts: "Epoch 1/150..."
```

**Estimated Effort:** 4-6 hours (section-by-section updates)

**Dependencies:** Task 1.1 (technical-specification rewrite) should be completed first for consistency

**Validation:**
- [ ] No references to `scripts/setup_kaggle.sh` or `scripts/run_training.sh`
- [ ] All code examples use Python notebook cell format
- [ ] No "SSH" or "tunnel" keywords (except in deprecation notices)
- [ ] Task dependencies updated to remove SSH prerequisite

---

## Phase 2: High Priority Documentation

### 2.1 Update `KAGGLE_TRAINING_GUIDE.md`

**Status:** MOSTLY CURRENT - Contains 1 critical SSH reference

**Issues Identified:**
- Line 60: "Vào notebook hiện tại của bạn (đang chạy SSH tunnel)" ← INCORRECT

**Required Changes:**

#### Line 60 (Step 1 Instruction)
**OLD:**
```markdown
1. Vào notebook hiện tại của bạn (đang chạy SSH tunnel)
```

**NEW:**
```markdown
1. Vào notebook hiện tại của bạn (Kaggle GPU Kernel với Internet + Secrets enabled)
```

#### Section: Prerequisites (Lines 1-30)
**Action:** ADD warning about deprecated SSH method

**INSERT at Line 15:**
```markdown
> ⚠️ **Important:** This guide describes the **Direct Notebook workflow** (current standard).
> The older SSH tunnel method is **deprecated** as of Dec 2024 due to GPU incompatibility.
> See `documentation/archive/deprecated-ssh-method/` for historical reference.
```

#### Section: Post-Training (Lines 200+)
**Action:** ADD Service Account limitation notice

**INSERT after training completion section:**
```markdown
### Known Limitation: DVC Push from Kaggle

**Issue:** DVC push fails from Kaggle with "Service Accounts do not have storage quota" (Error 403)

**Root Cause:** Google Drive API restriction - Service Accounts cannot write to personal "My Drive"

**Workaround (Accepted):**
1. Download trained model from Kaggle output (best.pt)
2. Run local DVC push: `dvc add weights/detection/best.pt && dvc push`
3. Commit .dvc file to Git

**Alternative (Requires Google Workspace):** Use Shared Drive instead of personal Drive
```

**Estimated Effort:** 1-2 hours

**Dependencies:** None

**Validation:**
- [ ] No SSH tunnel references
- [ ] Deprecation warning visible
- [ ] Service Account limitation documented

---

### 2.2 Update `kaggle-training-workflow.md`

**Status:** VERSION 2.0 - MOSTLY ACCURATE - Needs minor additions

**Issues Identified:**
- Contains comparison table (SSH vs Direct) - GOOD
- Missing Service Account limitation section

**Required Changes:**

#### Section: Known Limitations (NEW SECTION)
**Action:** ADD after "Workflow Comparison" section

**INSERT at Line ~150:**
```markdown
## Known Limitations & Workarounds

### DVC Push from Kaggle (Service Account)

**Limitation:** Service Accounts cannot upload to personal Google Drive (403 error)

**Impact:**
- `dvc push` from Kaggle notebook fails
- Trained models must be manually downloaded

**Workaround:**
```bash
# On Kaggle (after training):
# 1. Download output: Notebook → Output → Download All

# On local machine:
cd container-id-research
# 2. Extract best.pt to weights/detection/
cp ~/Downloads/best.pt weights/detection/

# 3. Version with DVC
dvc add weights/detection/best.pt
git add weights/detection/best.pt.dvc .gitignore
git commit -m "feat(detection): add model from exp001"

# 4. Push to DVC remote
dvc push

# 5. Push metadata to Git
git push
```

**Alternative Solution:** Use Google Workspace Shared Drive (enterprise feature)

**References:**
- Google Drive API: [Service Account Limitations](https://developers.google.com/drive/api/guides/about-service-accounts)
- GitHub Issue: #42 (DVC Push 403 Error)
```

#### Section: Version History (End of File)
**Action:** ADD Version 2.1 entry

**APPEND:**
```markdown
## Version History

### Version 2.1 (2024-12-09)
- Added: Service Account DVC push limitation documentation
- Added: Manual download workaround steps
- Updated: Known issues section

### Version 2.0 (2024-12-08)
- Migration from SSH tunnel to Direct Notebook workflow
- Added workflow comparison table
- Documented deprecation reasons

### Version 1.0 (2024-11-20)
- Initial SSH tunnel workflow documentation
```

**Estimated Effort:** 1 hour

**Dependencies:** None

**Validation:**
- [ ] Service Account limitation clearly documented
- [ ] Workaround steps tested and accurate
- [ ] Version history updated

---

### 2.3 Update `KAGGLE_SECRETS_SETUP.md`

**Status:** ✅ COMPLETED - File deleted, content merged into KAGGLE_TRAINING_GUIDE.md (Dec 11, 2024)

**Issues Identified:**
- Correctly documents DVC Service Account setup
- Missing warning about write limitation to personal Drive

**Required Changes:**

#### Section: Step 2 - DVC Service Account (Lines 50-100)
**Action:** ADD warning callout after share instruction

**INSERT at Line ~80 (after "Share Google Drive folder" step):**
```markdown
> ⚠️ **Important Limitation:** Service Accounts can **read** from personal Google Drive but **cannot write**.
> 
> **Impact:**
> - `dvc pull` works correctly ✅
> - `dvc push` fails with 403 error ❌
> 
> **Workaround:** Download models from Kaggle, then run `dvc push` locally.
> 
> **Alternative:** Use Google Workspace Shared Drive (requires paid Google Workspace account).
> 
> **Technical Details:** Google Drive API restricts Service Account write access to personal "My Drive".
> See: [Service Account Documentation](https://developers.google.com/drive/api/guides/about-service-accounts#permissions)
```

**Estimated Effort:** 30 minutes

**Dependencies:** None

**Validation:**
- [ ] Warning clearly visible
- [ ] Impact explained (pull works, push fails)
- [ ] Workaround documented
- [ ] Link to Google documentation included

---

## Phase 3: Medium Priority Documentation

### 3.1 Update `CHANGELOG.md`

**Status:** CURRENT (last updated Dec 9) - Needs Dec 9 entry for notebook commit

**Required Changes:**

#### Section: [Unreleased]
**Action:** ADD entry for kaggle_training_notebook.py commit (4240761)

**INSERT at top of Unreleased section:**
```markdown
### 🚀 Features
- **HIGH**: Added automatic DVC & Git sync to `kaggle_training_notebook.py` (Step 9)
  - Dynamic output detection: `weights/detection/weights/`, `weights/detection/`, `runs/`
  - Track only `best.pt` (skip epoch checkpoints for storage optimization)
  - Git authentication with GITHUB_TOKEN (OAuth format)
  - DVC push to Google Drive (note: fails for Service Accounts on personal Drive)
- **MEDIUM**: Committed kaggle_training_notebook.py with complete training workflow

### 🐛 Bug Fixes
- **MEDIUM**: Fixed indentation error in kaggle_training_notebook.py (line 809: `if ret == 0:`)
  - Changed from 20 spaces to 16 spaces for correct scope

### 📝 Documentation
- **HIGH**: Documented Service Account DVC push limitation (403 error)
  - Root cause: Google Drive API restriction on personal Drive write access
  - Workaround: Manual download + local `dvc push`
```

**Estimated Effort:** 30 minutes

**Dependencies:** None

**Validation:**
- [ ] Recent commits documented
- [ ] Service Account limitation mentioned
- [ ] Indentation fix recorded

---

### 3.2 Add Deprecation Notice to Root README

**Status:** UNKNOWN (need to check if mentions training workflow)

**Required Changes:**

#### Section: Module 1 Training (if exists)
**Action:** ADD deprecation notice if README references SSH workflow

**Example:**
```markdown
## Training Module 1 (Container Door Detection)

> 📌 **Current Workflow:** Direct Notebook (GitHub clone method)
> 🗄️ **Deprecated:** SSH tunnel method (archived Dec 2024)

See: [Kaggle Training Guide](KAGGLE_TRAINING_GUIDE.md)
```

**Estimated Effort:** 15 minutes

**Dependencies:** Need to read README.md first to assess if changes needed

**Validation:**
- [ ] No SSH references in README (or marked deprecated)
- [ ] Link to current training guide

---

## Phase 4: Code Cleanup

### 4.1 Mark Deprecated Scripts

**Status:** 2 scripts deprecated but not clearly marked

**Files:**
1. `scripts/setup_kaggle.sh` - Already has deprecation comment (Lines 2-5) ✅
2. `scripts/run_training.sh` - Missing deprecation notice ❌

**Required Changes:**

#### File: `scripts/run_training.sh`
**Action:** ADD deprecation header (matching setup_kaggle.sh format)

**INSERT at Line 2:**
```bash
#!/bin/bash
# ⚠️  DEPRECATED: SSH Tunnel Method No Longer Supported (as of Dec 2024)
# ⚠️  Reason: GPU driver incompatibility, environment complexity
# ⚠️  Use Instead: kaggle_training_notebook.py (direct notebook workflow)
# ⚠️  See: documentation/modules/module-1-detection/kaggle-training-workflow.md
#
# Complete Training Pipeline for Kaggle (SSH Workflow - LEGACY)
# Usage: bash scripts/run_training.sh [experiment_name]
# Context: Run inside SSH session on Kaggle VM (NOT RECOMMENDED)
```

**Estimated Effort:** 15 minutes

**Dependencies:** None

**Validation:**
- [ ] Both scripts have matching deprecation notices
- [ ] Reference to new workflow included
- [ ] Clear "NOT RECOMMENDED" warning

---

### 4.2 Consider Script Removal (OPTIONAL)

**Recommendation:** KEEP deprecated scripts for now (archive later)

**Rationale:**
- Scripts still functional for local testing/debugging
- May be useful reference for custom workflows
- Archive can be done in future cleanup sprint

**Alternative Action:** Move to archive directory
```bash
mkdir -p documentation/archive/deprecated-ssh-method/scripts
git mv scripts/setup_kaggle.sh documentation/archive/deprecated-ssh-method/scripts/
git mv scripts/run_training.sh documentation/archive/deprecated-ssh-method/scripts/
```

**Decision:** DEFER to user preference

---

## Phase 5: Validation & Testing

### 5.1 Documentation Link Validation

**Objective:** Ensure no broken links after refactoring

**Checklist:**
- [ ] All links to `kaggle_training_notebook.py` resolve correctly
- [ ] Links to deprecated SSH method point to archive directory
- [ ] Cross-references between docs updated (technical-spec ↔ implementation-plan)
- [ ] README links correct

**Tool:** Use VS Code "Find All References" or `grep -r "setup_kaggle.sh" documentation/`

**Estimated Effort:** 1 hour

---

### 5.2 Search for Remaining SSH References

**Objective:** Ensure all SSH/tunnel references removed or marked deprecated

**Command:**
```bash
# Search for SSH references across all markdown files
grep -rin "ssh\|tunnel\|ngrok\|cloudflared" documentation/ --include="*.md" | grep -v archive | grep -v "deprecated"
```

**Expected Result:** Only intentional references (e.g., in deprecation notices)

**Action:** Review each match, update or mark as deprecated

**Estimated Effort:** 1 hour

---

### 5.3 Test Current Workflow Documentation

**Objective:** Verify documentation matches actual execution

**Test Steps:**
1. Follow `KAGGLE_TRAINING_GUIDE.md` step-by-step on Kaggle
2. Verify kaggle_training_notebook.py executes without errors
3. Check all Kaggle Secrets configured correctly (DVC, GitHub, WandB)
4. Confirm dataset fetches successfully
5. Validate manual download workflow (post-training)

**Expected Outcome:** 
- Training completes successfully
- Model downloadable from Kaggle output
- Local `dvc push` works

**Estimated Effort:** 30 minutes (documentation reading) + 3 hours (Kaggle execution)

---

## Implementation Timeline

### Priority Matrix

| Phase         | Priority   | Files                               | Effort | Dependencies |
| ------------- | ---------- | ----------------------------------- | ------ | ------------ |
| **Phase 1.1** | CRITICAL   | technical-specification-training.md | 6-8h   | None         |
| **Phase 1.2** | CRITICAL   | implementation-plan.md              | 4-6h   | Phase 1.1    |
| **Phase 2.1** | HIGH       | KAGGLE_TRAINING_GUIDE.md            | 1-2h   | None         |
| **Phase 2.2** | HIGH       | kaggle-training-workflow.md         | 1h     | None         |
| **Phase 2.3** | ✅ OBSOLETE | KAGGLE_SECRETS_SETUP.md (deleted)   | 0min   | N/A          |
| **Phase 3.1** | MEDIUM     | CHANGELOG.md                        | 30min  | None         |
| **Phase 3.2** | MEDIUM     | README.md (if needed)               | 15min  | None         |
| **Phase 4.1** | LOW        | run_training.sh                     | 15min  | None         |
| **Phase 5**   | VALIDATION | All files                           | 2-3h   | All phases   |

### Recommended Execution Order

**Sprint 1 (Day 1-2): Critical Docs**
1. Task 1.1: Rewrite technical-specification-training.md (6-8h)
2. Task 1.2: Update implementation-plan.md (4-6h)
3. Commit: `docs(detection): rewrite technical specs for direct notebook workflow`

**Sprint 2 (Day 3): High Priority**
4. Task 2.1: Update KAGGLE_TRAINING_GUIDE.md (1-2h) ✅
5. Task 2.2: Update kaggle-training-workflow.md (1h) ✅
6. Task 2.3: ~~Update KAGGLE_SECRETS_SETUP.md~~ → File consolidated (deleted) ✅
7. Commit: `docs(detection): consolidate Kaggle documentation (remove 3 redundant files)` ✅

**Sprint 3 (Day 4): Cleanup & Validation**
8. Task 3.1: Update CHANGELOG.md (30min)
9. Task 3.2: Check README.md (15min)
10. Task 4.1: Mark run_training.sh deprecated (15min)
11. Phase 5: Validation (2-3h)
12. Commit: `docs(detection): complete refactoring for direct notebook workflow`

**Total Estimated Time:** 16-24 hours over 3-4 days

---

## Success Criteria

### Documentation Quality
- [ ] Zero SSH/tunnel references (except in archive and deprecation notices)
- [ ] All code examples use notebook cell format (no bash scripts)
- [ ] Architecture diagrams reflect Direct Notebook workflow
- [ ] Service Account limitation clearly documented

### Code Quality
- [ ] Deprecated scripts marked with clear warnings
- [ ] No broken links between documentation files
- [ ] CHANGELOG.md up-to-date

### Functional Validation
- [ ] Current workflow documentation tested on Kaggle
- [ ] Manual model download workflow verified
- [ ] All Kaggle Secrets setup instructions accurate

### User Experience
- [ ] New users can follow docs without confusion
- [ ] Clear migration path documented (old → new workflow)
- [ ] Archive directory properly organized

---

## Risk Assessment

### Low Risk Changes
✅ Adding warnings/notices to existing files (Phase 2.3, 3.1, 4.1)
✅ Updating CHANGELOG.md (Phase 3.1)

### Medium Risk Changes
⚠️ Rewriting large sections of implementation-plan.md (Phase 1.2)
⚠️ Updating workflow diagrams (Phase 1.1)

### High Risk Changes
🔴 Complete rewrite of technical-specification-training.md (Phase 1.1)
- **Mitigation:** Create backup, review diff carefully before commit
- **Validation:** Cross-check with kaggle_training_notebook.py implementation

---

## Rollback Plan

If refactoring introduces errors or confusion:

1. **Git Revert:** Use Git history to restore previous versions
   ```bash
   git log --oneline documentation/modules/module-1-detection/
   git revert <commit_hash>
   ```

2. **Incremental Commits:** Each phase committed separately for granular rollback

3. **Archive Preservation:** Original SSH documentation preserved in archive

---

## Post-Refactoring Maintenance

### Future Workflow Changes
When updating training workflow in future:

1. **Code First:** Update `kaggle_training_notebook.py`
2. **Docs Immediately:** Update documentation in same PR
3. **Changelog:** Document changes in CHANGELOG.md
4. **Version:** Bump version in workflow documentation

### Documentation Review Cadence
- Review training docs every 2 months
- Update after major workflow changes
- Check for broken links quarterly

---

## Appendix: Search Commands

### Find All SSH References
```bash
grep -rin "ssh\|tunnel\|ngrok\|cloudflared" documentation/ --include="*.md"
```

### Find All Script References
```bash
grep -rin "setup_kaggle.sh\|run_training.sh" documentation/ --include="*.md"
```

### Find All Deprecated Workflow Mentions
```bash
grep -rin "Phase 0:\|SSH session\|SSH terminal" documentation/modules/module-1-detection/ --include="*.md" | grep -v archive
```

---

## References

- **Current Workflow:** `kaggle_training_notebook.py`
- **Deprecation Notice:** `documentation/archive/deprecated-ssh-method/README.md`
- **Workflow Comparison:** `documentation/modules/module-1-detection/kaggle-training-workflow.md`
- **Service Account Issue:** GitHub Issue #42 (hypothetical - adjust if real issue exists)

---

**Approval Required:** Yes  
**Estimated Completion Date:** 2024-12-13 (4 days from creation)  
**Contact:** Project Maintainer  
**Status:** DRAFT - Pending Review
