from pathlib import Path

def vprint(text: str, config: dict) -> None:
    """Conditional printing based on verbosity defined in config file."""
    if config.get('verbose'):
        print(text)


def is_docker():
    """Check if the current environment is a Docker container."""
    cgroup = Path('/proc/self/cgroup')
    return Path('/.dockerenv').is_file() or (cgroup.is_file() and 'docker' in cgroup.read_text())