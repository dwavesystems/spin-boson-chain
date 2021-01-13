#!/usr/bin/env python
"""A module for retrieving the version of this library.

The version is provided in the standard python format ``major.minor.revision``
as a string. Use ``pkg_resources.parse_version`` to compare different versions.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# To get version of Python being used in the current (virtual) environment.
import sys

# To run git commands and retrieve corresponding output.
import subprocess

# For getting the absolute file path of this file.
import os



############################
## Authorship information ##
############################

__author__     = "Matthew Fitzpatrick"
__copyright__  = "Copyright 2021"
__credits__    = ["Matthew Fitzpatrick"]
__maintainer__ = "Matthew Fitzpatrick"
__email__      = "mfitzpatrick@dwavesys.com"
__status__     = "Development"



####################################
## Define functions and constants ##
####################################

# List of public objects in objects.
__all__ = ["version",
           "released",
           "short_version",
           "git_revision",
           "full_version",
           "version_summary"]



# Hard-coded version for people without git. Current non-production version.
version = '0.1.0'

# Whether this is a released version or modified.
released = False

# Short version.
short_version = 'v' + version



def _get_git_revision():
    """Get revision hash of ``libtemplate`` from git.

    Parameters
    ----------

    Returns
    -------
    revision : `str`
        Git revision hash of ``libtemplate``.
    """
    try:
        parsed_cmd = ['git', 'rev-parse', 'HEAD']
        cwd = os.path.dirname(os.path.abspath(__file__))
        stderr = subprocess.STDOUT

        cmd_output = subprocess.check_output(parsed_cmd, cwd=cwd, stderr=stderr)
        revision = cmd_output.decode().strip()

    except:
        revision = "unknown"
        
    return revision

# The current git revision (if available).
git_revision = _get_git_revision()



def _get_full_version():
    """Get version of ``libtemplate`` from git.

    Parameters
    ----------

    Returns
    -------
    full_version : `str`
        Full version of ``libtemplate``.
    """
    full_version = version
    
    if not released:
        full_version += '.dev0+' + git_revision[:7]
        
    return full_version

# Full version string including a prefix containing the beginning of the git
# revision hash.
full_version = _get_full_version()



def _get_version_summary():
    """Get version summary of ``libtemplate``.

    Parameters
    ----------

    Returns
    -------
    summary : `str`
        Version summary of ``libtemplate``.
    """
    # Check versions of libtemplate.
    from . import _version
    if _version.version != version:
        raise ValueError("libtemplate version has changed since "
                         "installation/compilation")



    # Generate summary.
    summary = ("libtemplate {libtemplate_ver!s};\n"
               "git revision {git_rev!s} using\n"
               "python {python_ver!s}")
    
    summary = summary.format(libtemplate_ver=full_version,
                             git_rev=git_revision,
                             python_ver=sys.version)
    return summary

# Summary of the versions as a string.
version_summary = _get_version_summary()
