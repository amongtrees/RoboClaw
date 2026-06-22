"""Simple dependency injection container for component wiring."""

from __future__ import annotations

from typing import Any


class ComponentRegistry:
    """IoC container for RoboClaw framework components.

    All major components (planners, executors, memory stores, etc.) register
    themselves here. The registry supports singleton and factory registration.

    Usage:
        registry = ComponentRegistry()
        registry.register(AbstractPlanner, TaskDecomposer(llm=...))
        planner = registry.get(AbstractPlanner)
    """

    def __init__(self) -> None:
        self._instances: dict[type, Any] = {}
        self._factories: dict[type, Any] = {}
        self._named: dict[str, Any] = {}

    def register(self, interface: type, implementation: Any) -> None:
        """Register a singleton implementation for an interface."""
        self._instances[interface] = implementation

    def register_factory(self, interface: type, factory: Any) -> None:
        """Register a factory callable for an interface (called on each get)."""
        self._factories[interface] = factory

    def register_named(self, name: str, component: Any) -> None:
        """Register a component by name."""
        self._named[name] = component

    def get(self, interface: type) -> Any:
        """Get the registered implementation for an interface.

        Returns the singleton if registered, otherwise calls the factory.
        Raises KeyError if neither is registered.
        """
        if interface in self._instances:
            return self._instances[interface]
        if interface in self._factories:
            instance = self._factories[interface]()
            self._instances[interface] = instance
            return instance
        raise KeyError(f"No implementation registered for {interface.__name__}")

    def get_named(self, name: str) -> Any:
        """Get a component by name."""
        if name not in self._named:
            raise KeyError(f"No component named '{name}' registered")
        return self._named[name]

    def has(self, interface: type) -> bool:
        """Check if an implementation is registered for an interface."""
        return interface in self._instances or interface in self._factories

    def reset(self) -> None:
        """Clear all registered components (useful for testing)."""
        self._instances.clear()
        self._factories.clear()
        self._named.clear()


# Global singleton registry
_registry: ComponentRegistry | None = None


def get_registry() -> ComponentRegistry:
    """Get or create the global component registry."""
    global _registry
    if _registry is None:
        _registry = ComponentRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _registry
    _registry = None
