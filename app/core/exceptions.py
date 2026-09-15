from app.shared.enums import FeatureTypeEnum


class NotFoundError(Exception):
    def __init__(self, resource: str, identifier: str | None = None) -> None:
        detail = f"{resource} not found"
        if identifier:
            detail = f"{resource} '{identifier}' not found"
        super().__init__(detail)
        self.detail = detail


class UnauthorizedError(Exception):
    def __init__(self, detail: str = "Unauthorized") -> None:
        super().__init__(detail)
        self.detail = detail


class ForbiddenError(Exception):
    def __init__(self, detail: str = "Access forbidden") -> None:
        super().__init__(detail)
        self.detail = detail


class ConflictError(Exception):
    def __init__(self, detail: str = "Resource already exists") -> None:
        super().__init__(detail)
        self.detail = detail


class QuotaExceededError(Exception):
    def __init__(self, feature: FeatureTypeEnum, limit: int, used: int) -> None:
        detail = f"Daily quota exceeded for '{feature}': {used}/{limit} used"
        super().__init__(detail)
        self.detail = detail
        self.feature = feature
        self.limit = limit
        self.used = used


class RateLimitedError(Exception):
    """Batidas demais no mesmo IP para a mesma rota (TCC-091).

    Vira 429 com ``Retry-After``. Separado de ``QuotaExceededError``, que também
    é 429 mas significa outra coisa: aquele é a cota do plano do usuário, este é
    o freio anônimo contra força bruta.
    """

    def __init__(self, retry_after: int) -> None:
        detail = "Muitas tentativas. Espere um pouco e tente de novo."
        super().__init__(detail)
        self.detail = detail
        self.retry_after = retry_after


class InferenceUnavailableError(Exception):
    """A classificação real não pôde ser executada (UX-001).

    Vira 503. Existe porque o comportamento anterior era pior que um erro: com
    o ONNX carregado, qualquer falha de decodificação ou de sessão caía no mock,
    que devolve uma doença **aleatória** com 70–95% de confiança. Na prática o
    sistema afirmava um diagnóstico inventado com a mesma cara de um real — o
    risco mais grave para uma demonstração ou para quem decide manejo no campo.

    Com ONNX disponível, o service agora ou classifica de verdade ou levanta
    isto. O mock segue existindo apenas quando nenhum modelo foi carregado
    (flag desligada ou arquivos ausentes), e nesse caso o resultado vem
    marcado com ``simulated=True``.
    """

    def __init__(self, detail: str = "Não foi possível analisar a imagem.") -> None:
        super().__init__(detail)
        self.detail = detail


class EmailDeliveryError(Exception):
    """O provedor transacional não aceitou ou não alcançou o envio.

    A rota transforma esta exceção em 503 para que o frontend não confirme um
    e-mail que nunca foi entregue. O detalhe é deliberadamente genérico: não
    expõe credenciais, respostas do provedor ou informações do destinatário.
    """

    def __init__(self, detail: str = "O serviço de e-mail está indisponível.") -> None:
        super().__init__(detail)
        self.detail = detail
