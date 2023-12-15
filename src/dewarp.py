import cv2
import numpy as np

GRID_CELLS_H = 20
GRID_CELLS_V = 10


def dewarp_page(img: np.ndarray, contour: np.ndarray, **kwargs) -> np.ndarray:
    """
    Dewarp and crop the image using the contours.
    :param img: Image to dewarp (not modified in place).
    :param contour: Contour points in image, in *clockwise order* (this is important).
    :return: Dewarped image
    """
    corners = find_corners(contour)
    # +1 here because to have N cells we need N+1 points.
    edges = interpolate_edges(contour, corners, nh=GRID_CELLS_H + 1, nv=GRID_CELLS_V + 1)

    src_grid = create_source_grid(edges)
    dst_grid = create_destination_grid(edges)

    img_dewarp = transform_grid(img, src_grid, dst_grid)

    # Debug
    if "intermediates" in kwargs:
        img_contour = np.copy(img)
        grid = src_grid
        for i in range(GRID_CELLS_V + 1):
            for j in range(GRID_CELLS_H + 1):
                if j < GRID_CELLS_H:
                    cv2.line(img_contour, tuple(grid[i, j, :]), tuple(grid[i, j + 1, :]), (0, 255, 0), thickness=10)
                if i < GRID_CELLS_V:
                    cv2.line(img_contour, tuple(grid[i, j, :]), tuple(grid[i + 1, j, :]), (0, 255, 0), thickness=10)

        intermediates = kwargs["intermediates"]
        intermediates["img_dewarp_grid"] = img_contour

    return img_dewarp


def find_corners(contour: np.ndarray) -> np.ndarray:
    """
    Find the top-left, top-right, bottom-right and bottom-left points.
    :param points: Points to find the extremities from.
    :return: array of the 4 extremities indices: top left, top right, bottom right, bottom left
    """
    s = contour.sum(axis=1)
    diff = np.diff(contour, axis=1)

    corners = np.zeros(4, dtype=np.int32)
    corners[0] = np.argmin(s)
    corners[2] = np.argmax(s)
    corners[1] = np.argmin(diff)
    corners[3] = np.argmax(diff)

    return corners


def interpolate_edges(contour: np.ndarray, corners: np.ndarray, nh: int, nv: int) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Using corner positions, interpolate points on the edges of the contour.
    The horizontal edges are returned in left to right order.
    The vertical edges are returned in top to bottom order.
    :param contour: List of contour points along page, in clockwise order.
    :param corners: Indices of corners in the contour array (in usual order).
    :param nh: Number of points along horizontal edges.
    :param nh: Number of points along vertical edges.
    :return: Tuple of edge arrays (top, right, bottom, left).
    """
    # Reorder contour point so that top left point is first,
    # so that edges can be indexed continously.
    contour = np.roll(contour, -corners[0], axis=0)
    corners = (corners - corners[0]) % contour.shape[0]

    # Top edge
    ci = corners[0]
    cni = corners[1]
    x = np.linspace(contour[ci, 0], contour[cni, 0], nh)
    pts = contour[ci:cni + 1, :]
    top = np.zeros((nh, 2))
    top[:, 0] = x
    top[:, 1] = np.interp(x, pts[:, 0], pts[:, 1])

    # Right edge
    ci = corners[1]
    cni = corners[2]
    y = np.linspace(contour[ci, 1], contour[cni, 1], nv)
    pts = contour[ci:cni + 1, :]
    right = np.zeros((nv, 2))
    right[:, 0] = np.interp(y, pts[:, 1], pts[:, 0])
    right[:, 1] = y

    # Bottom edge
    ci = corners[2]
    cni = corners[3]
    x = np.linspace(contour[cni, 0], contour[ci, 0], nh)
    pts = contour[ci:cni + 1, :][::-1]  # Reverted since we want left to right
    bottom = np.zeros((nh, 2))
    bottom[:, 0] = x
    bottom[:, 1] = np.interp(x, pts[:, 0], pts[:, 1])

    # Left edge (reverted)
    ci = corners[3]
    cni = corners[0]
    y = np.linspace(contour[cni, 1], contour[ci, 1], nv)
    pts = np.roll(contour, -1, axis=0)[ci - 1:, :][::-1]  # Reverted since we want top to bottom
    # Roll by -1 to put first corner at last index to also include it.
    left = np.zeros((nv, 2))
    left[:, 0] = np.interp(y, pts[:, 1], pts[:, 0])
    left[:, 1] = y

    return top, right, bottom, left


def create_source_grid(edges: tuple) -> np.ndarray:
    """
    Create a grid from page edge points.
    The grid is created using top and bottom edges since we assume curvature is mostly along the horizontal axis.
    :param edges: Edges returned from interpolate_edges.
    :return: NxMx2 grid of points, where N is the size of horizontal edges and M the size of vertical edges.
    """
    top, right, bottom, left = edges
    nh = top.shape[0]
    nv = right.shape[0]
    grid = np.zeros((nv, nh, 2))

    # Left and right edges are already known, no need to interpolate
    grid[:, 0, :] = left
    grid[:, -1, :] = right

    # Interpolate the rest of the grid, using horizontal edges
    for i in range(1, nh - 1):
        yp = top[i, 1], bottom[i, 1]
        y = np.linspace(*yp, nv)
        x = np.interp(y, yp, (top[i, 0], bottom[i, 0]))
        grid[:, i, 0] = x
        grid[:, i, 1] = y

    return np.int32(np.round(grid))


def create_destination_grid(edges: tuple) -> np.ndarray:
    """
    Create a destination grid from page edge points.
    The grid is created by approximating page dimensions from the edge lengths.
    :param edges: Edges returned from interpolate_edges.
    :return: NxMx2 grid of points, where N is the size of horizontal edges and M the size of vertical edges.
    """
    nh = edges[0].shape[0]
    nv = edges[1].shape[0]

    # Compute edge lengths and use it to choose the destination grid dimensions
    # Edge length is computed as the sum of the distance between all of its points
    lengths = [np.sum(np.sqrt(np.sum(np.diff(e, axis=0) ** 2, axis=1))) for e in edges]
    width = 0.5 * (lengths[0] + lengths[2])
    height = 0.5 * (lengths[1] + lengths[3])

    # Create grid from coordinate axii
    x = np.linspace(0, width, nh)
    y = np.linspace(0, height, nv)
    xx, yy = np.meshgrid(x, y)
    grid = np.concatenate((xx[:, :, np.newaxis], yy[:, :, np.newaxis]), axis=2)

    return np.int32(np.round(grid))


def transform_grid(img: np.ndarray, src_grid: np.ndarray, dst_grid: np.ndarray) -> np.ndarray:
    """
    Transform source grid in image to destination grid, creating a new image.
    Note: this only works for an orthogonal destination grid.
    :param img: Source image to transform
    :param src_grid: Source grid points (integers)
    :param dst_grid: Destination grid points (integers), same shape as src_grid.
    :return: Transformed image
    """

    def get_grid_quad(grid, i, j):
        # Return 4 points array for the cell (i, j) in a grid.
        DIFF = ((0, 0), (1, 0), (0, 1), (1, 1))
        return np.float32([tuple(grid[i + dy, j + dx]) for dx, dy in DIFF])

    result = np.zeros((*dst_grid[-1, -1, :][::-1], 3), dtype=np.uint8)
    for i in range(src_grid.shape[0] - 1):
        for j in range(src_grid.shape[1] - 1):
            src_quad = get_grid_quad(src_grid, i, j)
            dst_quad = get_grid_quad(dst_grid, i, j)

            # Do perspective transform.
            # Put result in shifted quad to avoid allocating whole image size for each cell.
            dst_quad_shift = dst_quad - dst_quad[0, :]
            M = cv2.getPerspectiveTransform(src_quad, dst_quad_shift)
            cell_size = tuple(np.int32(dst_quad_shift[3, :]))
            cell = cv2.warpPerspective(img, M, cell_size)

            # Transfer dewarped cell to result image.
            dst_tl = np.int32(dst_quad[0, :])
            dst_br = np.int32(dst_quad[3, :])
            result[dst_tl[1]:dst_br[1], dst_tl[0]:dst_br[0]] = cell

    return result
