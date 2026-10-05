from dataclasses import dataclass

from fastapi import HTTPException

from app.organizations.assistant import read_assistant_settings
from app.organizations.service import normalize_site_host, organization_for_saved_host


@dataclass(frozen=True)
class SiteRegistration:
    registered: bool
    organization_name: str
    portal_enabled: bool = True
    memory_enabled: bool = True
    profile_api_path: str = ""


def registration_for_host(site_host: str) -> SiteRegistration:
    try:
        cleaned_host = normalize_site_host(site_host)
    except HTTPException:
        return SiteRegistration(registered=False, organization_name="")
    organization = organization_for_saved_host(cleaned_host)
    if organization is None:
        return SiteRegistration(registered=False, organization_name="")
    settings = read_assistant_settings(organization.id)
    return SiteRegistration(
        registered=True,
        organization_name=organization.name,
        portal_enabled=settings.portal_enabled,
        memory_enabled=settings.memory_enabled,
        profile_api_path=settings.profile_api_path,
    )
