# Architecture Specification

## Overview
The `aituber-talk` project is evolving from a collection of loosely coupled scripts into a host-driven platform that can orchestrate multiple presentation and conversation modes. The core goals are:
- Centralize lifecycle management, shared resources, and external interfaces in a controller service.
- Treat each execution flow (talk, present, future AI agent, etc.) as a pluggable mode that can be added, removed, or distributed independently.
- Provide a single web UI and API surface for session control, status monitoring, and input delivery.
- Consolidate dependency management so all components can share a consistent Python environment while keeping optional extras isolated.

## Layered Breakdown

### Controller Layer (`controller/`)
*Responsibilities*
- Load configuration, initialize shared dependencies, and boot the platform.
- Host REST and WebSocket/SSE endpoints (e.g., `/modes`, `/sessions`, `/sessions/{id}/events`).
- Manage session state, event dispatch, and dependency injection for modes.
- Discover and register mode plugins via entry points or configuration.

*Key Modules*
- `app.py`: Entry point for the controller service; wires dependencies and starts the ASGI server.
- `api.py`: Defines HTTP and real-time endpoints exposed to clients.
- `session.py`: Session manager that tracks lifecycle, status, and metadata.
- `events.py`: Event bus abstraction that delivers notifications from modes to connected clients.
- `registry.py`: Plugin registry that locates and instantiates available modes.

### Core Services (`core/`)
*Responsibilities*
- Provide reusable building blocks consumed by controller and plugin layers.

*Key Modules*
- `talk_engine.py`: Encapsulates the SadTalker pipeline, motion/audio queues, OSC client, blink logic, and playback coordination. Exposes a class-based API for synchronous and asynchronous speech tasks.
- `agent_runtime.py` (future): Houses LLM invocation, tool management, and conversational memory primitives shared by AI agent modes.
- `power_point.py`: Wraps `PowerPointController` utilities with a mode-friendly interface.
- `config.py`: Loads configuration files, environment variables, and default settings.

### Plugin Modes (`plugins/`)
*Responsibilities*
- Implement specific behaviors on top of the controller + core services.
- Communicate exclusively through the mode interface to keep the host decoupled.

*Mode Interface*
Each plugin implements the `Mode` abstract base class:
- `start(session: SessionContext, **params)`
- `pause(session_id)` / `resume(session_id)` / `stop(session_id)`
- `status(session_id)`
- Optional hooks (e.g., `configure`, `handle_event`) as the API evolves.

*Initial Plugins*
- `plugins.talk`: Migrates the existing `/talk` FastAPI logic. Receives text inputs from the controller, calls `TalkEngine`, and emits playback events.
- `plugins.present`: Ports `present_and_talk.py`. Controls PowerPoint slides, extracts script/imagery, and queues speech directly through `TalkEngine` without HTTP.
- `plugins.agent` (future): Implements an interactive conversation agent backed by `AgentRuntime`, enabling tool use and multi-turn dialogues.

Plugins may live in this repository or in external packages, provided they declare an entry point such as:
```toml
[project.entry-points."aituber_talk.modes"]
"talk" = "plugins.talk:TalkMode"
"present" = "plugins.present:PresentMode"
```

### UI Layer (`ui/web/`)
*Responsibilities*
- Interact with the controller API to create sessions, send inputs, and visualize events.
- Remain backend-agnostic: any mode-specific render logic is driven by metadata in events.

*Implementation Notes*
- The first iteration can remain a Gradio app (`ui/web/app.py`).
- Support mode selection, live session timeline, status indicators, and control actions (pause/resume/stop).
- Future migrations to React or other frameworks should reuse the same REST/WebSocket contract.

## Dependency & Environment Strategy
- Introduce a project-level `pyproject.toml` that defines the base package and optional extras.
- Suggested extras: `talk`, `present`, `agent`, `webui`. For example, `present` can include `pywin32`, while `agent` lists LLM-related libraries such as `langgraph`.
- Replace multiple ad-hoc virtual environments with a single environment (`python -m venv .venv` or conda) plus extras. Developers install only what they need via `pip install .[talk,present,webui]`.
- Maintain compatibility with conda by providing an `environment.yaml` that uses the same extras through the pip section.
- Legacy setup scripts (`23_install_webui.ps1`, etc.) are deprecated once the unified environment lands.

## Event Model
- Define canonical event types in `events/schema.py` (e.g., `speech_started`, `speech_completed`, `slide_changed`, `agent_prompt`, `agent_response`).
- Modes publish events through the controller's event bus; the UI subscribes via WebSocket/SSE.
- Metadata-rich payloads enable clients to render detailed views without coupling to internal implementations.

## Roadmap Alignment
1. **Extract Talk Engine**: Move SadTalker orchestration and queues into `core/talk_engine.py`; adapt the existing FastAPI endpoints to call the engine through the controller.
2. **Bootstrap Controller Service**: Implement the registry, session manager, event bus, and REST/WebSocket interface. Register the talk mode as the first plugin.
3. **Port Presentation Mode**: Convert `present_and_talk` into `plugins.present`, using direct engine calls and emitting slide/audio events.
4. **Revamp Web UI**: Update the Gradio app (or a new frontend) to interact with the controller endpoints. Provide a unified control surface for all modes.
5. **Unify Environments**: Ship `pyproject.toml`, define extras, and document the new setup workflow. Sunset legacy venv scripts.
6. **Agent Foundations**: Create `core/agent_runtime.py` scaffolding and a placeholder `plugins.agent` to prepare for conversational AITuber features.
7. **Testing & Distribution**: Add integration tests for mode lifecycle and event propagation. Support optional distribution of plugins as separate packages via entry points.

## Future Considerations
- **Distributed Execution**: The mode interface allows modes to run out-of-process (e.g., via IPC or HTTP) if necessary. Implement adapters that proxy mode calls to remote workers when workloads grow.
- **Persistence**: Sessions and events can be persisted to Redis, PostgreSQL, or other stores by swapping implementations in the controller layer.
- **Security & Auth**: Introduce authentication middleware and role-based access when exposing the controller service beyond local development.
- **Observability**: Standardize structured logging and metrics collection (e.g., OpenTelemetry exporters) within the controller for easier monitoring.

This specification serves as the reference for ongoing development. New code and refactors should align with these layering principles, plugin contracts, and environment practices.
