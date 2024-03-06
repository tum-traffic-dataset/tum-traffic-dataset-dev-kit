import math

import numpy as np
import dataclasses as dc
import json
from pathlib import Path
from zipfile import ZipFile
import cv2
from typing import List, Iterable, Dict, Optional, Tuple
from collections import defaultdict
import tqdm

from src.utils.detection import Detection
from src.utils.perspective import Perspective

"""
A lane section is uniquely identified by Road ID and S-Offset.
"""
LaneSectionID = Tuple[int, float]

"""
A lane is uniquely identified by LaneSection ID and Lane ID. 
"""
LaneID = Tuple[LaneSectionID, int]


def _get_transform_map2local(frame="road"):
    """Obtain transformation matrix from map to local frame. Code adopted from drawing library."""
    if frame == "road":
        rotation = np.array(
            [
                [-0.28401534, -0.95881973, 0.0],
                [
                    0.95881973,
                    -0.28401534,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                ],
            ]
        )
        translation = np.array([-7.67, -25.89, 0.0])
    elif frame == "s110_base":
        rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation = np.array([-854.96568588, -631.98486299, 0.0])
    else:
        raise NotImplementedError(f"Unknown coordinate frame '{frame}'!")
    transformation = np.zeros((4, 4))
    transformation[0:3, 0:3] = rotation
    transformation[0:3, 3] = translation
    transformation[3, 3] = 1.0
    return transformation


def _transform(matrix, pts):
    """Transform the given 3d points with a 4x4 matrix."""
    result = np.ones((len(pts), 4))
    result[:, :3] = pts
    result = np.matmul(matrix, result.reshape((-1, 4, 1))).reshape((-1, 4))[:, :3]
    return result


def _range_isect(a1, b1, a2, b2):
    """1-D Range intersection test"""
    return a2 <= a1 <= b2 or a2 <= b1 <= b2 or a1 <= a2 <= b1 or a1 <= b2 <= b1


def _point_line_side(a, b, c):
    """
    Computes the determinant of a matrix given by three vec2s A, B, C:

      |(C.x - A.x) (C.y - A.y)|
      |(B.x - A.x) (B.y - A.y)|

    The return value is the barycentric coordinate of C with respect to A.
    Its sign is useful to determine whether c is left or right of the line A->B.
    """
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])


# The map has some glitched lane line geometry.
HEADING_OVERRIDES = {
    (3132600, 15, 0): -3.0720254235341553,
    (3132600, 15, 1): -3.0720254235341553,
    (3132600, 17, 0): -3.0720254235341553,
}


@dc.dataclass
class LaneSection:
    """
    Processed lane section, annotated with a bounding box and transformed
    from map to road coordinate system.
    """

    id: LaneSectionID
    aabb: Optional[np.ndarray] = None  # Min and Max-points
    lanes: List[np.ndarray] = dc.field(default_factory=list)  # Sequence of shape-points for each lane

    def set_aabb(self, min_pt, max_pt):
        self.aabb = np.array([min_pt, max_pt, [min_pt[0], max_pt[1]], [max_pt[0], min_pt[1]]])

    def contains(self, point: np.ndarray):
        result = self.aabb is None or (
                np.all(np.less_equal(self.aabb[0], point[:2])) and np.all(np.less_equal(point[:2], self.aabb[1]))
        )
        return result

    def intersects(self, min_pos: np.ndarray, max_pos: np.ndarray):
        assert self.aabb is not None
        return _range_isect(min_pos[0], max_pos[0], self.aabb[0][0], self.aabb[1][0]) and _range_isect(
            min_pos[1], max_pos[1], self.aabb[0][1], self.aabb[1][1]
        )

    def crop_to_area(self, min_pos: np.ndarray, max_pos: np.ndarray):
        """
        Crop the lane section to the given parameter range
        """
        result = LaneSection(self.id)
        for lane in self.lanes:
            lane = lane[np.logical_and(lane[:, 0] >= min_pos[0], lane[:, 1] <= max_pos[0])]
            if len(lane) > 0:
                result.lanes.append(lane)
        if len(result.lanes) > 0:
            return result
        else:
            return None


def map_json():
    """Obtain path to the default lane_samples.json ZIP."""
    map_path = Path(__file__).parent / "map.zip"
    with ZipFile(map_path, "r") as map_container:
        with map_container.open("lane_samples.json") as map_file:
            return json.load(map_file)


def load_map_for_local_frame(local_frame, zero_plane=True) -> List[LaneSection]:
    """
    Load map and transform it to the coordinate frame
    indicated by `local_frame`. Currently allowed values are 'road' or 's110_base'.
    """
    lanes_json = map_json()
    transformation_matrix = _get_transform_map2local(local_frame)
    result: List[LaneSection] = []
    for road in lanes_json["roads"]:
        for lane_section in road["laneSections"]:
            converted_lane_section = LaneSection((road["road"], lane_section["laneSection"]))
            min_pts = []
            max_pts = []
            for lane in lane_section["lanes"]:
                # Transform from map to road coordinate system
                vertices = _transform(transformation_matrix, np.array(lane["samples"]))
                if zero_plane:
                    vertices[:, 2] = 0.0
                if len(vertices) > 1:
                    min_pts.append(np.min(vertices, axis=0)[:2])
                    max_pts.append(np.max(vertices, axis=0)[:2])
                    converted_lane_section.lanes.append(vertices)
            if converted_lane_section.lanes:
                converted_lane_section.set_aabb(np.min(min_pts, axis=0), np.max(max_pts, axis=0))
                result.append(converted_lane_section)
    return result


class HeadingGridSet:
    """Class which wraps a set of grids containing rendered heading values
    for an area in relation to a camera perspective."""

    def __init__(
            self,
            perspective: Perspective,
            lanes: List[LaneSection],
            resolution: float = 0.1,
            max_cam_distance: float = 200.0,
            filter_by_fov: bool = True,
    ):
        # TODO: on the highway the max_cam_distance should be 1000 meters
        # Determine visible lane sections per road ID
        self.resolution = resolution
        visible_lane_sections: Dict[int, List[LaneSection]] = defaultdict(list)
        for lane_section in lanes:
            pos = np.hstack([lane_section.aabb, np.zeros((4, 1))]).T
            dist = np.linalg.norm((pos - perspective.translation).T, axis=0)
            if np.any(dist > max_cam_distance):
                continue
            if filter_by_fov:
                height, width = perspective.image_shape
                image_pts = perspective.project_from_base_to_image(pos, filter_behind=True).T
                # The lane section is visible if any of its corner points are visible.
                is_visible = np.any(
                    np.logical_and(
                        np.all(np.greater_equal(image_pts, [0, 0]), axis=1),
                        np.all(np.less(image_pts, [width, height]), axis=1),
                    )
                )
                if is_visible:
                    visible_lane_sections[lane_section.id[0]].append(lane_section)
            else:
                visible_lane_sections[lane_section.id[0]].append(lane_section)
        # Determine required grid extent
        visible_ls_corners = np.concatenate([ls.aabb for lls in visible_lane_sections.values() for ls in lls])
        self.grid_min = self.snap_to_grid(np.min(visible_ls_corners, axis=0))
        self.grid_max = self.snap_to_grid(np.max(visible_ls_corners, axis=0))
        self.grid_extent = self.grid_max - self.grid_min
        self.grid_size = np.rint(self.grid_extent / self.resolution).astype(int) + np.ones(2, dtype=int)
        # Render required lane sections per road
        self.grids = np.zeros((len(visible_lane_sections), self.grid_size[0], self.grid_size[1], 2))
        self.road_ids = []
        for i, (road_id, road_lane_sections) in tqdm.tqdm(list(enumerate(visible_lane_sections.items()))):
            self.road_ids.append(road_id)
            grid_for_road = self.grids[i]
            for ls in road_lane_sections:
                self.render_lane_section(ls, grid_for_road)

    def snap_to_grid(self, pos: np.ndarray, force_inside=False):
        """Snap the given point to the nearest grid cell centroid away from zero.
        Optionally, make sure that the returned position is inside (self.min_pos, self.max_pos)."""
        result = (np.ceil(np.abs(pos[:2]) / self.resolution) + 0.5) * self.resolution * np.sign(pos[:2])
        if force_inside:
            result = np.minimum(self.grid_max, np.maximum(self.grid_min, result))
        return result

    def render_lane_section(self, lane_section: LaneSection, dest_buf: np.ndarray):
        """Render the triangles of one lane section into a specific buffer."""
        assert np.all(dest_buf.shape[0:2] == self.grid_size) and dest_buf.shape[2] == 2
        # Annotate lane lines with heading values
        heading_values_per_lane = []
        for i, lane in enumerate(lane_section.lanes):
            assert len(lane) > 1
            if i > 0:
                prev_lane = lane_section.lanes[i - 1]
                assert len(lane) == len(prev_lane)
            lane_section_lane_id = (lane_section.id[0], int(lane_section.id[1]), i)
            if lane_section_lane_id in HEADING_OVERRIDES:
                heading_values = np.array([HEADING_OVERRIDES[lane_section_lane_id]] * len(lane))
            else:
                heading_values = lane[1:] - lane[:-1]
                heading_values = np.concatenate([[heading_values[0]], heading_values])
                heading_values = np.arctan2(heading_values[:, 1], heading_values[:, 0])
                assert len(heading_values) == len(lane)
            heading_values_per_lane.append(heading_values)
        lane_vertices_with_heading = np.concatenate(
            [
                np.array(lane_section.lanes)[:, :, :2],
                np.array(heading_values_per_lane).reshape((len(lane_section.lanes), -1, 1)),
            ],
            axis=2,
        )
        # Gather lane triangles
        prev_lane = lane_vertices_with_heading[0]
        triangles = []
        for lane in lane_vertices_with_heading[1:]:
            assert len(lane) == len(prev_lane)
            for i in range(1, len(lane)):
                triangles.append(np.array([lane[i - 1], lane[i], prev_lane[i - 1]]))
                triangles.append(np.array([prev_lane[i - 1], lane[i], prev_lane[i]]))
            prev_lane = lane
        # Render lane triangles
        for tri in triangles:
            # Determine extent
            tri_grid_min = self.snap_to_grid(np.min(tri, axis=0), True)
            tri_grid_max = self.snap_to_grid(np.max(tri, axis=0), True)
            tri_grid_size = np.rint((tri_grid_max - tri_grid_min) / self.resolution)
            tri_grid_size = tri_grid_size.astype(int) + np.ones(2, dtype=int)
            # Apply world position values
            pixel_pos = (tri_grid_min + np.array(list(np.ndindex(*tri_grid_size))) * self.resolution).T
            # Gather pixel_inside and heading_and_confidence (via w1/w2/w3 barycentric triangle coordinates)
            tri_area = _point_line_side(tri[0], tri[1], tri[2])
            if tri_area == 0.0:
                continue
            w1 = _point_line_side(tri[0], tri[1], pixel_pos) / tri_area
            w2 = _point_line_side(tri[1], tri[2], pixel_pos) / tri_area
            w3 = _point_line_side(tri[2], tri[0], pixel_pos) / tri_area
            side = np.array([np.greater_equal(w1, 0), np.greater_equal(w2, 0), np.greater_equal(w3, 0)]).T
            pixel_inside = np.all(side, axis=1).reshape(tri_grid_size)
            heading_and_confidence = np.concatenate(
                [(w1 * tri[0][2] + w2 * tri[1][2] + w3 * tri[2][2]).reshape((-1, 1)), np.ones((len(w1), 1))], axis=1
            ).reshape((tri_grid_size[0], tri_grid_size[1], 2))
            # Set confidence and heading in result grid where pixel_inside
            grid_pos_in_result = ((tri_grid_min - self.grid_min) / self.resolution + 0.5).astype(int)
            grid_max_in_result = grid_pos_in_result + tri_grid_size
            dest_buf[grid_pos_in_result[0]: grid_max_in_result[0], grid_pos_in_result[1]: grid_max_in_result[1]][
                pixel_inside
            ] = heading_and_confidence[pixel_inside]

    def store_image(self, where: Path):
        """Store a visual representation of all heading grids."""
        where.parent.mkdir(exist_ok=True)
        result = np.zeros((self.grid_size[0], self.grid_size[1], 4))
        # For each grid, gather pixels which have a value
        # and blend them together into a single image.
        for i, grid in enumerate(self.grids):
            pixels_with_data = np.not_equal(grid[:, :, 1], 0.0)
            # -- Store heading color
            result[pixels_with_data, 1] += (np.cos(grid[pixels_with_data, 0]) * 0.5 + 1.0) * 255
            result[pixels_with_data, 2] += (np.sin(grid[pixels_with_data, 0]) * 0.5 + 1.0) * 255
            result[pixels_with_data, 3] += 100
            # -- Store identity color
            # result[pixels_with_data, 0] += random.randint(0, 255)
            # result[pixels_with_data, 1] += random.randint(0, 255)
            # result[pixels_with_data, 2] += random.randint(0, 255)
            # result[pixels_with_data, 3] += 100
        cv2.imwrite(str(where), result)

    def lookup(self, pos: np.ndarray) -> Iterable[float]:
        """Lookup possible headings for a particular 3D position."""
        # Map position to grid position
        pos = pos.flatten()[:2] - self.grid_min
        pos /= self.resolution
        pos = np.rint(pos).astype(int)
        if pos[0] < 0 or pos[1] < 0 or pos[0] >= self.grid_size[0] or pos[1] >= self.grid_size[1]:
            return []
        # Actually lookup a 9-patch...
        candidates = self.grids[
                     :,
                     [np.maximum(0, pos[0] - 1), pos[0], np.minimum(self.grid_size[0] - 1, pos[0] + 1)],
                     [np.maximum(0, pos[1] - 1), pos[1], np.minimum(self.grid_size[1] - 1, pos[1] + 1)],
                     ]
        return set(candidates[candidates[:, :, 1] > 0][:, 0])

    def align(self, detection: Detection):
        """Align a detection to point in a direction allowed by the map."""
        yaw_options = self.lookup(detection.location)
        if not yaw_options:
            print(f"WARNING: No HD-map yaw option at {detection.location.flatten()}!")
            return
        detection.pick_yaw(np.array(list(yaw_options)))

    def lookup_histogram(self, pos: np.ndarray) -> Optional[List[Tuple[int, int, float]]]:
        """Lookup possible headings for a list of positions in [[x], [y]] format.
        Returns a histogram of the lookup results per road id. Each result tuple
        is a combination of road id, number of hits and the average heading along
        that road for the given positions."""
        # Map position to grid position
        pos -= self.grid_min.reshape((2, 1))
        pos /= self.resolution
        pos = np.rint(pos).astype(int)
        within_bounds = np.logical_and(
            np.logical_and(0 <= pos[0], pos[0] < self.grid_size[0]),
            np.logical_and(0 <= pos[1], pos[1] < self.grid_size[1]),
        )
        pos = pos[:, within_bounds]
        if not len(pos):
            return None
        results_per_grid = self.grids[:, pos[0], pos[1]]
        result_histogram = []
        for i, results in enumerate(results_per_grid):
            hits = results[results[:, 1] > 0]
            if not len(hits):
                continue
            hits = hits[:, 0]
            hit_clusters = [hits]
            for _ in range(2):
                # Separate up to three heading clusters, so we don't average discontinuous heading values.
                # This is the case, when two lane sections meet, which are digitized in opposite directions.
                last_cluster = hit_clusters[-1]
                if not len(last_cluster):
                    break
                outliers = np.abs(last_cluster - last_cluster[0]) > math.pi * 0.5
                hit_clusters = hit_clusters[:-1] + [last_cluster[np.logical_not(outliers)], last_cluster[outliers]]
            for cluster in hit_clusters:
                if len(cluster):
                    avg_heading = np.average(cluster)
                    result_histogram.append((self.road_ids[i], len(cluster), avg_heading))
        if not result_histogram:
            return None
        # Sort candidate headings by number of hits
        result_histogram.sort(key=lambda entry: entry[1])
        return result_histogram
