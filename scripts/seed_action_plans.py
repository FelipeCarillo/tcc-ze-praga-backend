"""
Seed script — populates action_plans, action_plan_sources, and subscription_plans.

Usage:
    uv run python scripts/seed_action_plans.py
"""

import asyncio
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import AsyncSessionLocal
from app.models.action_plan import ActionPlan
from app.models.action_plan_source import ActionPlanSource
from app.models.crop import Crop
from app.models.subscription_plan import SubscriptionPlan

# ── Subscription Plans ────────────────────────────────────────────────────────

SUBSCRIPTION_PLANS = [
    {
        "name": "free",
        "display_name": "Gratuito",
        "chat_daily_limit": 10,
        "inference_daily_limit": 5,
        "api_monthly_limit": 0,
    },
    {
        "name": "pro",
        "display_name": "Pro",
        "chat_daily_limit": None,
        "inference_daily_limit": None,
        "api_monthly_limit": 500,
    },
    {
        "name": "enterprise",
        "display_name": "Enterprise",
        "chat_daily_limit": None,
        "inference_daily_limit": None,
        "api_monthly_limit": None,
    },
]

# ── Disease Action Plans ──────────────────────────────────────────────────────
# Source: tcc-ze-praga-frontend/src/services/mock/mockData.js

ACTION_PLANS = [
    {
        "disease_id": "ferrugem-asiatica",
        "levels": {
            "essencial": [
                "Aplique fungicida triazol + estrobilurina imediatamente",
                "Monitore lavoura diariamente nas próximas 2 semanas",
                "Registre pontos de infestação com GPS",
                "Evite trânsito de máquinas na área infectada",
            ],
            "campo": [
                "Realize aplicação aérea se área superior a 50 hectares",
                "Implemente rotação de culturas na próxima safra",
                "Ajuste espaçamento para melhorar ventilação",
                "Monitore umidade relativa e temperatura diariamente",
                "Aplique segunda dose de fungicida em 14 dias",
                "Documente progresso da doença com fotografias",
            ],
            "especialista": [
                "Consulte engenheiro agrônomo para análise de resistência fúngica",
                "Solicite análise laboratorial para confirmar cepa do patógeno",
                "Avalie histórico de uso de fungicidas na área",
                "Considere programa de manejo integrado de doenças",
                "Implemente sistema de alerta baseado em dados climáticos",
                "Avalie impacto econômico e calcule limiar de dano econômico",
            ],
        },
        "sources": [
            {
                "name": "EMBRAPA Soja",
                "detail": "Tecnologias de produção de soja - Região Central do Brasil",
                "url": "https://www.embrapa.br/soja",
                "display_order": 0,
            },
            {
                "name": "CONAB",
                "detail": "Acompanhamento da Safra Brasileira de Grãos",
                "url": "https://www.conab.gov.br",
                "display_order": 1,
            },
        ],
    },
    {
        "disease_id": "mancha-alvo",
        "levels": {
            "essencial": [
                "Aplique fungicida específico para Corynespora cassiicola",
                "Remova folhas severamente infectadas",
                "Aumente intervalo entre irrigações",
                "Evite molhamento foliar no período noturno",
            ],
            "campo": [
                "Implemente sistema de irrigação localizada",
                "Realize monitoramento semanal com escala de notas",
                "Aplique silício foliar para fortalecer tecidos",
                "Avalie necessidade de segunda aplicação em 21 dias",
                "Mantenha registro de condições climáticas",
            ],
            "especialista": [
                "Analise resistência a fungicidas benzimidazóis",
                "Consulte literatura sobre variedades resistentes",
                "Implemente programa de monitoramento epidemiológico",
                "Avalie rotação de princípios ativos fungicidas",
            ],
        },
        "sources": [
            {
                "name": "EMBRAPA Soja",
                "detail": "Doenças da Soja — Mancha Alvo",
                "url": "https://www.embrapa.br/soja",
                "display_order": 0,
            },
        ],
    },
    {
        # ADR-0003: a antracnose saiu do catálogo junto com a troca do Digipathos
        # pelo ASDID (que não a cobre) e deu lugar à mancha-olho-de-rã. O seed de
        # doenças foi atualizado na época, este aqui não — resultado: um plano
        # órfão de antracnose e uma das 6 classes do modelo SEM plano de ação,
        # devolvendo "plano indisponível" para um diagnóstico correto.
        "disease_id": "mancha-olho-de-ra",
        "levels": {
            "essencial": [
                "Aplique fungicida com estrobilurina + triazol nas primeiras lesões",
                "Anote a cultivar do talhão — a resistência genética é o controle principal",
                "Monitore a cada 7-10 dias, olhando o terço médio da planta",
            ],
            "campo": [
                "Inicie o controle químico entre R1 e R3, ao detectar as primeiras lesões",
                "Escolha cultivares resistentes a Cercospora sojina na próxima safra",
                "Faça rotação com milho ou outra não hospedeira por 1-2 safras",
                "Evite semear sobre palhada de soja infectada sem decomposição adequada",
                "Programe a segunda aplicação em 14-21 dias, rotacionando o grupo químico",
            ],
            "especialista": [
                "Diferencie da mancha-alvo: lesão menor (1-5 mm) e sem anéis concêntricos",
                "Considere resistência a QoI (mutação G143A) — nunca use estrobilurina isolada",
                (
                    "Acompanhe o monitoramento regional de raças — a resistência "
                    "é governada por genes Rcs"
                ),
                (
                    "Garanta cobertura do terço médio e inferior do dossel "
                    "(150 L/ha, gotas finas a médias)"
                ),
            ],
        },
        "sources": [
            {
                "name": "EMBRAPA Soja",
                "detail": "Manejo de doenças foliares da soja — mancha olho-de-rã",
                "url": "https://www.embrapa.br/soja",
                "display_order": 0,
            },
            {
                "name": "Zhang, G. et al. (2018)",
                "detail": (
                    "Frogeye leaf spot of soybean: pathogen biology and QoI "
                    "resistance. Plant Disease, 102(12)"
                ),
                "display_order": 1,
            },
        ],
    },
    {
        "disease_id": "cercosporiose",
        "levels": {
            "essencial": [
                "Aplique fungicida triazol preventivamente",
                "Monitore plantas nos estádios R1-R2",
                "Ajuste densidade de plantio para reduzir umidade",
                "Documente incidência por talhão",
            ],
            "campo": [
                "Implemente programa de monitoramento com base em graus-dia",
                "Avalie resistência varietal no portfólio disponível",
                "Realize análise de solo para nutrição equilibrada",
                "Mantenha pH do solo entre 6.0 e 6.5",
            ],
            "especialista": [
                "Estude epidemiologia local da Cercospora kikuchii",
                "Avalie eficácia de fungicidas registrados",
                "Consulte ZAE (Zoneamento Agrícola de Risco Climático)",
                "Implemente modelo preditivo de risco de epidemia",
            ],
        },
        "sources": [
            {
                "name": "EMBRAPA Soja",
                "detail": "Cercosporiose da Soja — Manejo e Controle",
                "url": "https://www.embrapa.br/soja",
                "display_order": 0,
            },
        ],
    },
    {
        "disease_id": "mildio",
        "levels": {
            "essencial": [
                "Aplique fungicida sistêmico (metalaxil + mancozebe)",
                "Melhore ventilação entre plantas",
                "Reduza umidade relativa no dossel",
                "Monitore surgimento de novos focos",
            ],
            "campo": [
                "Implemente tratamento de sementes com fungicidas sistêmicos",
                "Avalie resistência genética das cultivares em uso",
                "Realize aplicações preventivas em períodos chuvosos",
                "Mantenha histórico de ocorrências na propriedade",
            ],
            "especialista": [
                "Identifique raça fisiológica de Peronospora manshurica",
                "Consulte catálogo de resistência varietal da EMBRAPA",
                "Avalie rotação de fungicidas para prevenir resistência",
                "Implemente sistema de alerta epidemiológico regional",
            ],
        },
        "sources": [
            {
                "name": "EMBRAPA Soja",
                "detail": "Míldio da Soja — Diagnóstico e Controle",
                "url": "https://www.embrapa.br/soja",
                "display_order": 0,
            },
        ],
    },
    {
        "disease_id": "saudavel",
        "levels": {
            "essencial": [
                "Continue o manejo preventivo atual",
                "Mantenha monitoramento regular da lavoura",
                "Realize análise de solo periódica",
                "Documente as boas práticas em uso",
            ],
            "campo": [
                "Implemente programa de monitoramento sistemático",
                "Avalie resistência das cultivares usadas",
                "Mantenha nutrição equilibrada da cultura",
                "Registre condições climáticas para histórico",
            ],
            "especialista": [
                "Consulte agrônomo para otimizar o sistema de produção",
                "Avalie novas cultivares de alta produtividade",
                "Implemente agricultura de precisão para gestão de insumos",
                "Desenvolva plano de manejo integrado preventivo",
            ],
        },
        "sources": [
            {
                "name": "EMBRAPA",
                "detail": "Boas Práticas Agrícolas — Guia de Manejo",
                "url": "https://www.embrapa.br",
                "display_order": 0,
            },
        ],
    },
]

# ── Seed functions ────────────────────────────────────────────────────────────


async def seed_subscription_plans(db: AsyncSession) -> None:
    print("Seeding subscription plans...")
    for plan_data in SUBSCRIPTION_PLANS:
        result = await db.execute(
            select(SubscriptionPlan).where(SubscriptionPlan.name == plan_data["name"])
        )
        existing = result.scalar_one_or_none()
        if existing:
            print(f"  ✓ Plan '{plan_data['name']}' already exists, skipping.")
            continue

        plan = SubscriptionPlan(**plan_data)
        db.add(plan)
        print(f"  + Created plan '{plan_data['name']}'")

    await db.commit()


async def seed_action_plans(db: AsyncSession) -> None:
    print("Seeding action plans...")
    # Todas as doencas do catalogo seedado pertencem a 'soja' (sprint A2).
    # Resolve o crop_id uma vez — ``ActionPlan.crop_id`` eh NOT NULL desde
    # a migration 0004_link_diagnoses_to_diseases.
    crop_result = await db.execute(select(Crop).where(Crop.slug == "soja"))
    soja_crop = crop_result.scalar_one_or_none()
    if soja_crop is None:
        raise RuntimeError(
            "Crop 'soja' nao encontrada — rode 'uv run python -m scripts.seed_crops' antes."
        )

    for disease in ACTION_PLANS:
        disease_id = disease["disease_id"]

        # Seed levels
        for level_name, actions in disease["levels"].items():
            result = await db.execute(
                select(ActionPlan).where(
                    ActionPlan.disease_id == disease_id,
                    ActionPlan.level == level_name,
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                print(f"  ✓ ActionPlan '{disease_id}/{level_name}' already exists, skipping.")
                continue

            db.add(
                ActionPlan(
                    crop_id=soja_crop.id,
                    disease_id=disease_id,
                    level=level_name,
                    actions=actions,
                )
            )
            print(f"  + Created action plan '{disease_id}/{level_name}'")

        # Seed sources (idempotent by disease_id + name)
        for source_data in disease["sources"]:
            result = await db.execute(
                select(ActionPlanSource).where(
                    ActionPlanSource.disease_id == disease_id,
                    ActionPlanSource.name == source_data["name"],
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                continue

            db.add(ActionPlanSource(disease_id=disease_id, **source_data))
            print(f"  + Created source '{source_data['name']}' for '{disease_id}'")

        await db.commit()


async def main() -> None:
    print("Starting seed...\n")
    async with AsyncSessionLocal() as db:
        await seed_subscription_plans(db)
        await seed_action_plans(db)
    print("\nSeed completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
