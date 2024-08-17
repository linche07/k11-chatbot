import random
import heapq

class MallLocator:
    def __init__(self):
        self.mall_graph = self._get_mall_graph()
        self.locations = self._get_locations()

    def _generate_random_coordinates(self):
        """Generate random latitude and longitude for a location."""
        lat = random.uniform(22.2930, 22.2950)  # Latitude range for the mall
        lon = random.uniform(114.1730, 114.1760)  # Longitude range for the mall
        return lat, lon

    def _get_mall_graph(self):
        """Randomly generate connections between locations within the mall."""
        floors = ['B2', 'B1', 'GF', '1F', '2F', '3F', '4F', '5F', '6F', '7F']
        graph = {}

        for floor in floors:
            locations = [f'{floor} A区', f'{floor} B区', f'{floor} C区']
            for loc in locations:
                # Randomly connect to other locations on the same floor
                connections = random.sample(locations, k=random.randint(1, 2))
                connections = [(conn, random.randint(5, 20)) for conn in connections if conn != loc]

                # Randomly connect to one location on another floor
                if random.random() > 0.5:
                    other_floor = random.choice([f for f in floors if f != floor])
                    connections.append((f'{other_floor} A区', random.randint(10, 30)))

                graph[loc] = connections

        return graph

    def _get_locations(self):
        """Generate random coordinates for each location in the mall."""
        floors = ['B2', 'B1', 'GF', '1F', '2F', '3F', '4F', '5F', '6F', '7F']
        locations = {}

        for floor in floors:
            for area in ['A区', 'B区', 'C区']:
                loc_name = f'{floor} {area}'
                locations[loc_name] = self._generate_random_coordinates()

        return locations

    def get_random_location(self):
        """Returns a random location from the mall."""
        location = random.choice(list(self.locations.keys()))
        lat, lon = self.locations[location]
        return location, lat, lon

    def dijkstra(self, start, end):
        """Dijkstra's algorithm to find the shortest path in the mall graph."""
        queue = [(0, start, [])]
        seen = set()
        while queue:
            (cost, node, path) = heapq.heappop(queue)
            if node in seen:
                continue

            path = path + [node]
            if node == end:
                return (cost, path)

            seen.add(node)
            for (next_node, distance) in self.mall_graph.get(node, []):
                heapq.heappush(queue, (cost + distance, next_node, path))
        return float("inf"), []

    def find_shortest_path(self, start_location, end_location):
        """Find and return the shortest path between two locations."""
        return self.dijkstra(start_location, end_location)
