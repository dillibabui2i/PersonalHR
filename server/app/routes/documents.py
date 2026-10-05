from fastapi import APIRouter, BackgroundTasks, File, Header, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from app.admin.session import require_organization_access
from app.documents.service import (
    DocumentRecord,
    delete_document,
    embed_document,
    list_documents,
    save_uploaded_document,
)

router = APIRouter()


class DocumentResponse(BaseModel):
    id: str
    file_name: str = Field(serialization_alias="fileName")
    status: str
    error_message: str = Field(serialization_alias="errorMessage")


class UploadBatchResponse(BaseModel):
    documents: list[DocumentResponse]
    errors: list[str]


def document_response(document: DocumentRecord) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        file_name=document.file_name,
        status=document.status,
        error_message=document.error_message,
    )


@router.get("/admin/organizations/{organization_id}/documents", response_model=list[DocumentResponse])
def read_documents(
    organization_id: str,
    x_admin_session: str = Header(default=""),
) -> list[DocumentResponse]:
    require_organization_access(x_admin_session, organization_id)
    return [document_response(document) for document in list_documents(organization_id)]


@router.post(
    "/admin/organizations/{organization_id}/documents",
    response_model=UploadBatchResponse,
    status_code=201,
)
async def add_documents(
    organization_id: str,
    background_tasks: BackgroundTasks,
    x_admin_session: str = Header(default=""),
    files: list[UploadFile] = File(),
) -> UploadBatchResponse:
    require_organization_access(x_admin_session, organization_id)
    if files == []:
        raise HTTPException(status_code=400, detail="Choose files to upload.")
    saved: list[DocumentResponse] = []
    errors: list[str] = []
    for file in files:
        label = file.filename or "file"
        try:
            if file.filename is None:
                raise HTTPException(status_code=400, detail="Choose a file to upload.")
            content = await file.read()
            document = save_uploaded_document(organization_id, file.filename, content)
        except HTTPException as error:
            detail = error.detail if isinstance(error.detail, str) else "The file could not be uploaded."
            errors.append(f"{label}: {detail}")
            continue
        background_tasks.add_task(embed_document, organization_id, document.id)
        saved.append(document_response(document))
    if saved == []:
        raise HTTPException(status_code=400, detail=" ".join(errors) or "Choose files to upload.")
    return UploadBatchResponse(documents=saved, errors=errors)


@router.delete("/admin/organizations/{organization_id}/documents/{document_id}", status_code=204)
def remove_document(
    organization_id: str,
    document_id: str,
    x_admin_session: str = Header(default=""),
) -> Response:
    require_organization_access(x_admin_session, organization_id)
    delete_document(organization_id, document_id)
    return Response(status_code=204)
