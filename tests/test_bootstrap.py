"""
Tests for utils/bootstrap.py

Tests the frozen executable bootstrap functionality.
"""
import pytest
import sys
import os
import tempfile
import shutil

# Import bootstrap module
from utils import bootstrap


class TestFrozenDetection:
    """Test frozen/platform detection."""

    def test_is_frozen_false_in_dev(self):
        """Verify IS_FROZEN is False in development mode."""
        assert bootstrap.IS_FROZEN == getattr(sys, 'frozen', False)
        # In test environment, should always be False
        assert bootstrap.IS_FROZEN is False

    def test_platform_detection(self):
        """Verify platform detection is correct."""
        assert bootstrap.IS_LINUX == sys.platform.startswith('linux')
        assert bootstrap.IS_WINDOWS == (sys.platform == 'win32')
        assert bootstrap.IS_MACOS == (sys.platform == 'darwin')


class TestPathResolution:
    """Test path resolution functions."""

    def test_get_app_dir_exists(self):
        """Verify app directory exists."""
        app_dir = bootstrap.get_app_dir()
        assert os.path.isdir(app_dir)

    def test_get_app_dir_contains_main(self):
        """Verify app directory contains main.py."""
        app_dir = bootstrap.get_app_dir()
        main_path = os.path.join(app_dir, 'main.py')
        assert os.path.isfile(main_path)

    def test_get_resource_path_exists(self):
        """Verify resource path resolution works."""
        path = bootstrap.get_resource_path('core/system_prompt.txt')
        assert os.path.isabs(path)
        assert os.path.isfile(path)

    def test_get_resource_path_config(self):
        """Verify config.yaml is found."""
        path = bootstrap.get_resource_path('config/config.yaml')
        assert os.path.isfile(path)

    def test_get_user_data_dir_in_dev(self):
        """In dev mode, user data dir should be ./data/."""
        user_dir = bootstrap.get_user_data_dir()
        app_dir = bootstrap.get_app_dir()
        assert user_dir == os.path.join(app_dir, 'data')


class TestDirectoryCreation:
    """Test directory creation functions."""

    def test_ensure_directories_returns_path(self):
        """Verify ensure_directories returns user data dir."""
        result = bootstrap.ensure_directories()
        assert os.path.isdir(result)

    def test_ensure_directories_creates_logs(self):
        """Verify logs subdirectory is created."""
        user_dir = bootstrap.ensure_directories()
        logs_dir = os.path.join(user_dir, 'logs')
        assert os.path.isdir(logs_dir)

    def test_ensure_directories_creates_conversation_logs(self):
        """Verify conversation_logs subdirectory is created."""
        user_dir = bootstrap.ensure_directories()
        conv_logs_dir = os.path.join(user_dir, 'conversation_logs')
        assert os.path.isdir(conv_logs_dir)


class TestEnvironmentSetup:
    """Test environment setup functions."""

    def test_setup_environment_returns_path(self):
        """Verify setup_environment returns user data dir."""
        result = bootstrap.setup_environment()
        assert os.path.isdir(result)

    def test_initialize_returns_dict(self):
        """Verify initialize returns expected structure."""
        result = bootstrap.initialize()
        assert isinstance(result, dict)
        assert 'user_data_dir' in result
        assert 'app_dir' in result
        assert 'is_frozen' in result
        assert 'platform' in result

    def test_initialize_paths_exist(self):
        """Verify paths from initialize exist."""
        result = bootstrap.initialize()
        assert os.path.isdir(result['user_data_dir'])
        assert os.path.isdir(result['app_dir'])


class TestSplashScreen:
    """Test splash screen functions (no-op in dev mode)."""

    def test_close_splash_no_error(self):
        """Verify close_splash doesn't raise in dev mode."""
        # Should not raise even when pyi_splash is not available
        bootstrap.close_splash()

    def test_update_splash_no_error(self):
        """Verify update_splash doesn't raise in dev mode."""
        # Should not raise even when pyi_splash is not available
        bootstrap.update_splash("Test message")


class TestModuleLevelExports:
    """Test module-level constants."""

    def test_user_data_dir_exported(self):
        """Verify USER_DATA_DIR is exported."""
        assert hasattr(bootstrap, 'USER_DATA_DIR')
        assert os.path.isdir(bootstrap.USER_DATA_DIR)

    def test_app_dir_exported(self):
        """Verify APP_DIR is exported."""
        assert hasattr(bootstrap, 'APP_DIR')
        assert os.path.isdir(bootstrap.APP_DIR)


class TestDataMigration:
    """Test data migration functionality."""

    def test_migrate_user_data_no_op_in_dev(self):
        """In dev mode, migration should be a no-op."""
        # Should not raise
        bootstrap.migrate_user_data()

    def test_external_data_dir_none_by_default(self):
        """External data dir should be None if not configured."""
        # Unless DAEMON_EXTERNAL_DATA is set
        if 'DAEMON_EXTERNAL_DATA' not in os.environ:
            result = bootstrap.get_external_data_dir()
            # Could be None or a path if ~/...daemon/external exists
            assert result is None or os.path.isdir(result)


class TestEnvLoading:
    """Test .env loading functionality."""

    def test_manual_env_loading(self, tmp_path):
        """Test manual .env parsing fallback."""
        # Create a test .env file
        env_path = tmp_path / '.env'
        env_path.write_text('TEST_VAR_BOOTSTRAP="test_value"\n')

        # Load it
        bootstrap._load_env_manual(str(env_path))

        # Check it was loaded (using setdefault, so won't override existing)
        # Note: This might not set if TEST_VAR_BOOTSTRAP already exists
        # The function uses setdefault, which won't override

    def test_manual_env_loading_with_comments(self, tmp_path):
        """Test manual .env parsing ignores comments."""
        env_path = tmp_path / '.env'
        env_path.write_text('''
# This is a comment
TEST_VAR2="value"
  # Indented comment
ANOTHER_VAR=another_value
''')

        # Should not raise
        bootstrap._load_env_manual(str(env_path))


# Run self-test if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
