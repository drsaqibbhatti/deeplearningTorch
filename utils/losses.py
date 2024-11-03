
################################ GIoU, DIoU, CIoU ############################

class BBCoordinates(NamedTuple):
    x1: torch.Tensor
    y1: torch.Tensor
    x2: torch.Tensor
    y2: torch.Tensor

    def get_width(self) -> torch.Tensor:
        return self.x2 - self.x1

    def get_height(self) -> torch.Tensor:
        return self.y2 - self.y1

    def get_area(self) -> torch.Tensor:
        return self.get_width() * self.get_height()
    
def intersection_area(
    boxes1: BBCoordinates,
    boxes2: BBCoordinates,
) -> torch.Tensor:
    x1, y1, x2, y2 = boxes1
    x1g, y1g, x2g, y2g = boxes2

    x1i = torch.max(x1, x1g)
    y1i = torch.max(y1, y1g)
    x2i = torch.min(x2, x2g)
    y2i = torch.min(y2, y2g)

    return (x2i - x1i).clamp(0) * (y2i - y1i).clamp(0)


def union_area(
    boxes1: BBCoordinates,
    boxes2: BBCoordinates,
    inter: torch.Tensor,
) -> torch.Tensor:
    return boxes1.get_area() + boxes2.get_area() - inter


def convex_width_height(
    boxes1: BBCoordinates,
    boxes2: BBCoordinates,
) -> tuple[torch.Tensor, torch.Tensor]:
    x1, y1, x2, y2 = boxes1
    x1g, y1g, x2g, y2g = boxes2

    cw = torch.max(x2, x2g) - torch.min(x1, x1g)  # convex width
    ch = torch.max(y2, y2g) - torch.min(y1, y1g)  # convex height

    return cw, ch


def compute_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    b1_coords = BBCoordinates(x1, y1, x2, y2)
    b2_coords = BBCoordinates(x1g, y1g, x2g, y2g)

    inter = intersection_area(b1_coords, b2_coords)
    union = union_area(b1_coords, b2_coords, inter)

    iou = inter / (union + eps)

    return iou


def iou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
    reduction: Literal["none", "mean", "sum"] = "none",
) -> torch.Tensor:
    iou = compute_iou(boxes1, boxes2, eps=eps)

    loss = 1 - iou

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()

    return loss


def giou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
    reduction: Literal["none", "mean", "sum"] = "none",
) -> torch.Tensor:
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    b1_coords = BBCoordinates(x1, y1, x2, y2)
    b2_coords = BBCoordinates(x1g, y1g, x2g, y2g)

    inter = intersection_area(b1_coords, b2_coords)
    union = union_area(b1_coords, b2_coords, inter)

    iou = inter / (union + eps)

    S = 1 - iou

    # compute the penality term

    cw, ch = convex_width_height(b1_coords, b2_coords)

    convex_area = cw * ch

    penality = torch.abs(convex_area - union) / torch.abs(convex_area + eps)

    loss = S + penality

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()

    return loss


def diou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
    reduction: Literal["none", "mean", "sum"] = "none",
) -> torch.Tensor:
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    b1_coords = BBCoordinates(x1, y1, x2, y2)
    b2_coords = BBCoordinates(x1g, y1g, x2g, y2g)

    iou = compute_iou(boxes1, boxes2, eps=eps)

    cw, ch = convex_width_height(b1_coords, b2_coords)

    # convex diagonal squared
    diagonal_distance_squared = cw**2 + ch**2

    # compute center distance squared
    b1_x = (x1 + x2) / 2
    b1_y = (y1 + y2) / 2
    b2_x = (x1g + x2g) / 2
    b2_y = (y1g + y2g) / 2

    centers_distance_squared = (b1_x - b2_x) ** 2 + (b1_y - b2_y) ** 2

    S = 1 - iou
    D = centers_distance_squared / (diagonal_distance_squared + eps)

    loss = S + D

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()

    return loss


def ciou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,     #  Avoid division by zero
    reduction: Literal["none", "mean", "sum"] = "none", # To reduce the computed loss from a batch of values to a single scalar value.
    # none: No reduction. The loss is returned as an unreduced tensor. The function will return a tensor with the same size as the input
    # sum: The loss is summed over all samples in the batch.
    # mean: the loss is mean over all samples in the batch
) -> torch.Tensor:
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    b1_coords = BBCoordinates(x1, y1, x2, y2)
    b2_coords = BBCoordinates(x1g, y1g, x2g, y2g)

    iou = compute_iou(boxes1, boxes2, eps=eps)

    ############## part 1 ###########
    S = 1 - iou

    cw, ch = convex_width_height(b1_coords, b2_coords)

    #The convex width (cw) is the distance between the minimum x1 (leftmost point) and the maximum x2 (rightmost point) of both boxes.
    # The convex height (ch) is the distance between the minimum y1 (topmost point) and the maximum y2 (bottom-most point) of both boxes.
   
    # convex diagonal squared
    diagonal_distance_squared = cw**2 + ch**2

    # compute center distance squared
    b1_x = (x1 + x2) / 2
    b1_y = (y1 + y2) / 2
    b2_x = (x1g + x2g) / 2
    b2_y = (y1g + y2g) / 2

    centers_distance_squared = (b1_x - b2_x) ** 2 + (b1_y - b2_y) ** 2

    ############## part 2 ##############
    D = centers_distance_squared / (diagonal_distance_squared + eps)

    w1, h1 = b1_coords.get_width(), b1_coords.get_height()
    w2, h2 = b2_coords.get_width(), b2_coords.get_height()

    v = (4 / math.pi**2) * torch.pow(
        torch.atan(w2 / h2) - torch.atan(w1 / h1),
        2,
    )

    with torch.no_grad():
        alpha = v / ((1 - iou) + v)

    ############# part 3 ###############
    V = alpha * v

    loss = S + D + V

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()

    return loss
###########################################################################################