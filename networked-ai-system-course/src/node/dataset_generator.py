from typing import List, Tuple
import sys, os
import numpy as np
from sklearn.datasets import make_circles, make_moons
from loguru import logger

# RANDOM_SEED = os.environ["RANDOM_SEED"]
RANDOM_SEED = 42


logger.add(sys.stderr, format="{time} - {level} - {message}", level="DEBUG")


class DatasetGenerator:
    def __init__(
        self,
        centers: List[List[float]] | None = None,
        stds: List[List[float]] | None = None,
    ) -> None:
        """Init method for dataset generator. Give initial center values

        Parameters
        ----------
        centers : List[List[float]]|None
            List of means to generate dataset distributions from. If None,
            then randomly set.
        stds : List[List[float]]|None
            List of stds to generate dataset distributions from. If None,
            then randomly set.
        """
        self.call_count = 0
        self.rng = np.random.default_rng(np.random.randint(0, 10000))
        self.rotation_proba = self.rng.uniform(low=0.2, high=0.7)
        self.centers = centers
        if self.centers is None:
            self.centers = [
                [(self.rng.random() + 1), (self.rng.random() + 1)],
                [(self.rng.random() + 4), (self.rng.random() + 2)],
                [(self.rng.random() - 1), (self.rng.random() + 3)],
            ]
        self.stds = stds
        if self.stds is None:
            self.stds = [
                self.rng.uniform(low=0.01, high=0.2),
                self.rng.uniform(low=0.01, high=0.2),
                self.rng.uniform(low=0.01, high=0.2),
            ]
        self.generator_founctions = [
            self.make_blobs,
            self.make_moons,
            self.make_circles,
        ]

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Call method that returns new datapoints

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Returns x, y why; the features of the new datapoints and the labels.
        """
        if self.rng.random() < self.rotation_proba:
            self.call_count += 1
        X = np.zeros(shape=(len(self.generator_founctions) * 1000, 2), dtype=np.float32)
        y = np.zeros(shape=(len(self.generator_founctions) * 1000), dtype=np.float32)
        for idx, generator_func in enumerate(self.generator_founctions):
            (
                X[idx * 1000 : (idx + 1) * 1000, :],
                y[idx * 1000 : (idx + 1) * 1000],
            ) = generator_func(self.centers[idx], self.stds[idx])

        def rotate(coords: List[float]) -> List[float]:
            """Rotates a x and y coordinates around z for self.call_count degrees.

            Parameters
            ----------
            coords : List[float]
                X,Y coordinate of a data point

            Returns
            -------
            List[float]
                Rotated coordinates.
            """
            f = lambda x, y: (
                np.sin(self.call_count * np.pi / 180) * y
                - np.cos(self.call_count * np.pi / 180) * x,
                np.sin(self.call_count * np.pi / 180) * x
                + np.cos(self.call_count * np.pi / 180) * y,
            )
            return f(coords[0], coords[1])

        X = np.apply_along_axis(rotate, axis=1, arr=X)
        return X, y

    def make_blobs(
        self, center: List[float], std: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns 1000 points from sklearn.datasets.make_blobs with the given center
        and standard deviation.

        Parameters
        ----------
        center : float
            x, and y coordinates of the center
        std : float
            Standard deviation of the pionts in each cluster

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            X,y of 1000 new data points

        """
        X = np.zeros(shape=(1000, 2))
        y = np.zeros(shape=(1000))
        X[:500, :] = self.rng.normal(loc=(0.5, 0.5), scale=std, size=(500, 2))
        y[:500] = 0
        X[500:, :] = self.rng.normal(loc=(-0.5, -0.5), scale=std, size=(500, 2))
        y[500:] = 1
        means = np.mean(X, axis=0)
        for idx, mean in enumerate(means):
            X[:, idx] += center[idx] - mean
        return X, y

    def make_circles(
        self, center: List[float], std: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns 1000 new datapoints in the shape of circles with the
        given center and std.

        Parameters
        ----------
        center : List[float]
            x, and y coordinates of the center
        std : float
            Noise in the data points

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            X,y of 1000 new data points
        """
        X, y = make_circles(n_samples=1000, noise=std)
        means = np.mean(X, axis=0)
        for idx, mean in enumerate(means):
            X[:, idx] += center[idx] - mean
        return X, y

    def make_moons(
        self, center: List[float], std: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns 1000 new datapoints in the shape of moons with the
        given center and std.

        Parameters
        ----------
        center : List[float]
            x, and y coordinates of the center
        std : float
            Noise in the data points

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            X,y of 1000 new data points
        """
        X, y = make_moons(n_samples=1000, noise=std)
        means = np.mean(X, axis=0)
        for idx, mean in enumerate(means):
            X[:, idx] += center[idx] - mean
        return X, y


if __name__ == "__main__":
    # Some test bed
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    subplot_titles = []
    for name in ["A", "B", "C"]:
        subplot_titles += (
            f"Dataset {name} after 1 iteration",
            f"Dataset {name} after 501 iteration",
            f"Dataset {name} after 1001 iteration",
        )
    fig = make_subplots(rows=3, cols=3, subplot_titles=subplot_titles,
                        horizontal_spacing = 0.05,
                        vertical_spacing = 0.1)
    np.random.seed(42)
    for r in range(1, 4):
        dg = DatasetGenerator()
        X, y = dg()
        logger.info(f"Data shape: {X.shape, y.shape}")
        fig.add_trace(
            go.Scatter(
                x=X[:, 0],
                y=X[:, 1],
                mode="markers",
                marker=dict(color=["#1f77b4" if c == 1 else "#ff7f0e" for c in y]),
            ),
            row=r,
            col=1,
        )
        for _ in range(500):
            X, y = dg()
        fig.add_trace(
            go.Scatter(
                x=X[:, 0],
                y=X[:, 1],
                mode="markers",
                marker=dict(color=["#1f77b4" if c == 1 else "#ff7f0e" for c in y]),
            ),
            row=r,
            col=2,
        )

        for _ in range(500):
            X, y = dg()
        fig.add_trace(
            go.Scatter(
                x=X[:, 0],
                y=X[:, 1],
                mode="markers",
                marker=dict(color=["#1f77b4" if c == 1 else "#ff7f0e" for c in y]),
            ),
            row=r,
            col=3,
        )
    fig.update_layout(margin=dict(t=0,b=0,r=0,l=0))
    fig.update_xaxes(range=[-7.5, 7.5], title="x1")
    fig.update_yaxes(range=[-7.5, 7.5], title="x2")
    fig.update_layout(template="none", showlegend=False,
                      width=1000,height=1000,
                      margin=dict(t=20,b=40,r=0,l=40))
    fig.show()
    fig.write_image("./networked-ai-system-course//figs/datset_image.svg")
