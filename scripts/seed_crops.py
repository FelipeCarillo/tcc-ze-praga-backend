"""Seed crops + diseases (initially soja + 6 doenças).

Idempotent: upsert by slug. Migrates the catalog from
``InferenceService._DISEASES`` into the database.

Usage:
    uv run python -m scripts.seed_crops
"""

import asyncio
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import AsyncSessionLocal
from app.models.crop import Crop
from app.models.disease import Disease
from app.shared.enums import SeverityEnum

# ── Crops ────────────────────────────────────────────────────────────────────

CROPS = [
    {
        "slug": "soja",
        "name_pt": "Soja",
        "scientific_name": "Glycine max",
        "kingdom": "Plantae",
    },
]

# ── Diseases (per crop slug) ─────────────────────────────────────────────────

DISEASES_BY_CROP: dict[str, list[dict]] = {
    "soja": [
        {
            "slug": "ferrugem-asiatica",
            "name_pt": "Ferrugem Asiática",
            "scientific_name": "Phakopsora pachyrhizi",
            "severity_default": SeverityEnum.ALTA.value,
            "description_md": (
                "Doença fúngica severa que causa lesões amareladas a marrom-escuras na face inferior "
                "das folhas. É a principal doença da soja no Brasil, podendo causar desfolha precoce "
                "e perdas de até 80% na produtividade."
            ),
        },
        {
            "slug": "mancha-alvo",
            "name_pt": "Mancha-Alvo",
            "scientific_name": "Corynespora cassiicola",
            "severity_default": SeverityEnum.MEDIA.value,
            "description_md": (
                "Doença fúngica caracterizada por lesões circulares com anéis concêntricos que lembram "
                "um alvo. Causa desfolha prematura e redução na produtividade, especialmente em "
                "cultivares suscetíveis."
            ),
        },
        {
            "slug": "antracnose",
            "name_pt": "Antracnose",
            "scientific_name": "Colletotrichum truncatum",
            "severity_default": SeverityEnum.MEDIA.value,
            "description_md": (
                "Doença fúngica que afeta hastes, vagens e sementes, causando lesões escuras e "
                "deprimidas. Pode causar morte de plântulas e apodrecimento de vagens, reduzindo "
                "a qualidade e quantidade dos grãos."
            ),
        },
        {
            "slug": "cercosporiose",
            "name_pt": "Cercosporiose",
            "scientific_name": "Cercospora kikuchii",
            "severity_default": SeverityEnum.BAIXA.value,
            "description_md": (
                "Doença fúngica que causa manchas púrpuras nas sementes (mancha púrpura) e lesões "
                "foliares. Reduz a qualidade das sementes e pode causar perdas moderadas na "
                "produtividade quando ocorre em condições favoráveis."
            ),
        },
        {
            "slug": "mildio",
            "name_pt": "Míldio",
            "scientific_name": "Peronospora manshurica",
            "severity_default": SeverityEnum.BAIXA.value,
            "description_md": (
                "Doença causada por oomiceto que produz lesões amareladas na face superior das folhas "
                "com esporulação cinza-esbranquiçada na face inferior. Geralmente causa perdas "
                "econômicas moderadas, exceto em condições de alta umidade."
            ),
        },
        {
            "slug": "saudavel",
            "name_pt": "Saudável",
            "scientific_name": None,
            "severity_default": SeverityEnum.NENHUMA.value,
            "description_md": (
                "A folha analisada não apresenta sinais visíveis de doenças fúngicas ou bacterianas. "
                "Continue monitorando regularmente para detecção precoce de possíveis infecções."
            ),
        },
    ],
}


# ── Seed functions ───────────────────────────────────────────────────────────


async def seed_crops(db: AsyncSession) -> dict[str, str]:
    """Upsert crops by slug. Returns slug -> crop_id mapping."""
    print("Seeding crops...")
    slug_to_id: dict[str, str] = {}
    for crop_data in CROPS:
        result = await db.execute(select(Crop).where(Crop.slug == crop_data["slug"]))
        existing = result.scalar_one_or_none()
        if existing:
            print(f"  ✓ Crop '{crop_data['slug']}' already exists, skipping.")
            slug_to_id[crop_data["slug"]] = existing.id
            continue

        crop = Crop(**crop_data)
        db.add(crop)
        await db.flush()
        slug_to_id[crop_data["slug"]] = crop.id
        print(f"  + Created crop '{crop_data['slug']}' (id={crop.id})")

    await db.commit()
    return slug_to_id


async def seed_diseases(db: AsyncSession, slug_to_crop_id: dict[str, str]) -> None:
    """Upsert diseases by (crop_id, slug)."""
    print("Seeding diseases...")
    for crop_slug, diseases in DISEASES_BY_CROP.items():
        crop_id = slug_to_crop_id.get(crop_slug)
        if not crop_id:
            print(f"  ! Crop '{crop_slug}' not found, skipping its diseases.")
            continue

        for disease_data in diseases:
            result = await db.execute(
                select(Disease).where(
                    Disease.crop_id == crop_id, Disease.slug == disease_data["slug"]
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                print(f"  ✓ Disease '{crop_slug}/{disease_data['slug']}' already exists, skipping.")
                continue

            db.add(Disease(crop_id=crop_id, **disease_data))
            print(f"  + Created disease '{crop_slug}/{disease_data['slug']}'")

    await db.commit()


async def main() -> None:
    print("Starting seed_crops...\n")
    async with AsyncSessionLocal() as db:
        slug_to_id = await seed_crops(db)
        await seed_diseases(db, slug_to_id)
    print("\nSeed completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
