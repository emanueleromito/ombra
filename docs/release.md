# Release

A release is one tag push. The workflow `.github/workflows/release.yml` checks that every version
reads the same, builds the wheels, builds and pushes the two images, and publishes the wheels.

## What a release publishes

| What | Where |
| --- | --- |
| `ombra`, `ombra-openfhe`, `langchain-ombra` wheels | PyPI |
| `ghcr.io/erailabs/ombra-server` (GPU, Ampere `sm_80` and `sm_86`) | GHCR |
| `ghcr.io/erailabs/ombra-cpu` (Python client with local mode, amd64 and arm64) | GHCR |

The CPU image has no `ombra_server`. The server needs CUDA and a GPU, so only the GPU image
serves search over gRPC. The CPU image is for local mode and tests.

Versions: `ombra`, `langchain-ombra`, the CMake project and `ombra_server --version` are 0.1.0.
`ombra-openfhe` stays at 1.6.1.0, because it follows openfhe-python and OpenFHE.

## To release

1. Set the same version in `client/pyproject.toml`, `client/ombra/__init__.py`,
   `integrations/langchain-ombra/pyproject.toml` and the top-level `CMakeLists.txt`. Add an entry
   to `CHANGELOG.md`.
2. Run the workflow by hand first: Actions, Release, Run workflow, with `dry_run` on. It publishes
   to TestPyPI and pushes the images as `dryrun-<commit>`.
3. Tag the release commit on `main` with `v<version>` and push the tag. The workflow refuses a tag
   that does not match the version in `client/pyproject.toml`, a tag whose commit is not on `main`,
   and any real release (`dry_run` off) that is not started from a `v*` tag. Both the image push
   and the PyPI upload then wait for your approval, in that order.

## One-time setup

Do this once, before the first run. No API token is stored anywhere: PyPI trusts the workflow
through OpenID Connect.

**GitHub environments.** In the repository settings, create four environments: `pypi`, `testpypi`,
`ghcr-release` and `ghcr-dryrun`. On `pypi` and `ghcr-release`, add yourself as a required reviewer
and restrict deployment to `v*` tags, so a tag push waits for your approval before it uploads wheels
or pushes the release image tags. Also add a tag ruleset that protects `v*`.

**Trusted publishers.** A project that does not exist yet is registered as a "pending publisher".
Do this on both https://pypi.org/manage/account/publishing/ and
https://test.pypi.org/manage/account/publishing/, once for each of `ombra`, `ombra-openfhe` and
`langchain-ombra`:

- Owner: `erailabs`
- Repository: `ombra`
- Workflow: `release.yml`
- Environment: `pypi` on pypi.org, `testpypi` on test.pypi.org

Test the whole chain on a fork or a test repository by changing the owner and repository.

**GHCR visibility.** The first push creates the packages `ombra-server` and `ombra-cpu` as private.
For each one, open the package settings on GitHub, link it to the repository, and set the
visibility to public, so `docker pull` works without a login. Check that the repository's workflow
permissions allow the `GITHUB_TOKEN` to write packages (the workflow asks for `packages: write`).

## What was not run in the cloud

The workflow and the images are written to be run by GitHub. The first dry run will show any
problem in them; expect the GPU image to take about an hour, since it builds OpenFHE and the CUDA
code.
