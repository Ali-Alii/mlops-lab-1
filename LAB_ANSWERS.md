# MLOps Lab Answers

## Q1. What does `uv init` do?

`uv init` initializes a new Python project. It creates the basic project
structure, including files such as `pyproject.toml`.

`pyproject.toml` contains information about the project and its Python
dependencies.

---

## Q2. What does `dvc init` do? Should the generated files be committed to Git?

`dvc init` initializes DVC in the Git repository. It creates the `.dvc/`
directory and `.dvcignore`.

The DVC configuration files should generally be committed to Git because other
developers need them to work with the project.

The **actual dataset should not be committed to Git**.

---

## Q3. Why shouldn't we push DVC credentials to GitHub? What does `--global` mean?

Credentials are secrets, so we should **never commit or push them to GitHub**.

`--global` means the DVC configuration applies to the **current user/machine**,
rather than only the current repository.

---

## Q4. What happens when we run `dvc add data`?

DVC starts tracking the `data/` directory. It:

1. Creates or updates `data.dvc`.
2. Adds `data/` to `.gitignore`.
3. Git therefore ignores the actual dataset.
4. Git tracks the small `data.dvc` file instead.
5. DVC manages the actual dataset.

In summary:

```text
data/       -> actual images    -> DVC manages it
data.dvc    -> pointer/metadata -> Git tracks it
```

---

## Q5. What is inside `data.dvc`?

`data.dvc` contains **metadata describing the dataset**, including information
such as its path and hash/checksum.

It **does not contain the actual images**.

Its purpose is to identify **which version of the data** the project is using.

---

## Q6. What is stored on GitHub and what is stored on DagsHub/DVC remote?

**GitHub:**

```text
Code
data.dvc
.dvc configuration
.gitignore
```

**DagsHub/DVC remote:**

```text
Actual dataset
```

> **GitHub stores the pointer; the DVC remote stores the actual data.**

---

## Q7. After cloning the repository, will the dataset automatically be available?

No. Git only gives you the code and the DVC pointer (`data.dvc`). You need to
run:

```bash
dvc pull
```

This downloads the actual dataset from the configured DVC remote/DagsHub.

---

## Q8. What happens when switching between commits and using `dvc checkout`?

Suppose:

```text
Commit A -> data.dvc -> DATA V1
Commit B -> data.dvc -> DATA V2
```

If you run:

```bash
git checkout A
```

Git changes the project files, including `data.dvc`, so it now points to
**DATA V1**. Then:

```bash
dvc checkout
```

DVC reads the pointer in `data.dvc` and makes the local data match the pointed
version:

```text
git checkout
     |
     v
changes data.dvc / pointer
     |
     v
dvc checkout
     |
     v
reads pointer and changes local data
     |
     v
local data = version referenced by data.dvc
```
