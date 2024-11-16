import numpy as np
import packcircles
from importlib import resources

from typing import List
from numpy.typing import ArrayLike

from PIL import Image, ImageDraw
from TetriumColor.Utils.CustomTypes import PlateColor, TetraColor


class IshiharaPlate:
    plates = [27, 35, 39, 64, 67, 68, 72, 73, 85, 87, 89, 96]

    def __init__(self, color: PlateColor, secret: int,
                 num_samples:int = 100, dot_sizes:List[int] = [16,22,28], image_size:int = 1024,
                 directory:str = '.', seed:int = 0, lum_noise:float =0, noise:float = 0, gradient:bool = False):
        """
        :param plate_color: A PlateColor object with shape and background colors (RGB/OCV tuples).

        :param secret:  May be either a string or integer, specifies which
                        secret file to use from the secrets directory.
        """
        self.inside_color = self.__standardize_color(color.shape)
        self.outside_color = self.__standardize_color(color.background)

        self.num_samples = num_samples
        self.dot_sizes = dot_sizes
        self.image_size = image_size
        self.directory = directory
        self.seed = seed
        self.noise = noise
        self.gradient = gradient
        self.lum_noise = lum_noise

        self.__set_secret_image(secret)

        self.__reset_plate()

    def __set_secret_image(self, secret:int):
        if secret in IshiharaPlate.plates:
            with resources.path("TetriumColor.Assets.HiddenImages", f"{str(secret)}.png") as data_path:
                self.secret = Image.open(data_path)
            self.secret = self.secret.resize([self.image_size, self.image_size])
            self.secret = np.asarray(self.secret)
        else:
            raise ValueError(f"Invalid Hidden Number {secret}")


    def generate_plate(self, seed: int = None, hiddenNumber:int = None, plate_color: PlateColor = None):
        """
        Generate the Ishihara Plate with specified inside/outside colors and secret.
        A new seed can be specified to generate a different plate pattern.
        New inside or outside colors may be specified to recolor the plate
        without modifying the geometry.

        :param seed: A seed for RNG when creating the plate pattern.
        :param inside_color: A 6-tuple RGBOCV color.
        :param outside_color: A 6-tuple RGBOCV color.
        """
        def helper_generate():
            self.__generate_geometry()
            self.__compute_inside_outside()
            self.__draw_plate()

        if plate_color:
            self.inside_color = self.__standardize_color(plate_color.shape)
            self.outside_color = self.__standardize_color(plate_color.background)
        
        if hiddenNumber:
            self.__set_secret_image(hiddenNumber)
        
        # Plate doesn't exist; set seed and colors and generate whole plate.
        if self.circles is None:
            self.seed = seed or self.seed
            helper_generate()
            return

        # Need to generate new geometry and re-color.
        if seed and seed != self.seed:
            self.seed = seed
            self.__reset_plate()
            helper_generate()
            return
        
        # Don't regenerate geometry but recolor w/new hidden number
        if not seed and hiddenNumber:
            self.__compute_inside_outside()
            self.__reset_images()
            self.__draw_plate()
            return

        # Need to re-color, but don't need to re-generate geometry or hidden number.
        if not seed and (plate_color):
            self.__reset_images()
            self.__draw_plate()
            return


    def export_plate(self, filenameRGB: str, filenameOCV: str):
        """
        This method saves two images - RGB and OCV encoded image.

        :param save_name: Name of directory to save plate to.
        :param ext: File extension to use, such as 'png' or 'tif'.
        """

        self.channels[0].save(filenameRGB)
        self.channels[1].save(filenameOCV)


    def __standardize_color(self, color: TetraColor):
        """
        :param color: Ensure a TetraColor is a float in [0, 1].
        """
        if np.issubdtype(color.RGB.dtype, np.integer):
            color.RGB = color.RGB.astype(float) / 255.0
        
        if np.issubdtype(color.OCV.dtype, np.integer):
            color.OCV = color.OCV.astype(float) / 255.0
        try:
            np.concatenate([color.RGB, color.OCV])
        except:
            import pdb; pdb.set_trace()
        return np.concatenate([color.RGB, color.OCV])


    def __generate_geometry(self):
        """
        :return output_circles: List of [x, y, r] sequences, where (x, y)
                                are the center coordinates of a circle and r
                                is the radius.
        """
        np.random.seed(self.seed)

        # Create packed_circles, a list of (x, y, r) tuples.
        radii = self.dot_sizes * 2000
        np.random.shuffle(radii)
        packed_circles = packcircles.pack(radii)

        # Generate output_circles.
        center = self.image_size // 2
        output_circles = []

        for (x, y, radius) in packed_circles:
            if np.sqrt((x - center) ** 2 + (y - center) ** 2) < center * 0.95:
                r = radius - np.random.randint(2, 5)
                output_circles.append([x,y,r])

        self.circles = output_circles


    def __compute_inside_outside(self):
        """
        For each circle, estimate the proportion of its area that is inside or outside.
        Take num_sample point samples within each circle, generated by rejection sampling.
        """
        # Inside corresponds to numbers; outside corresponds to background
        outside = np.int32(np.sum(self.secret == 255, -1) == 4)
        inside  = np.int32((self.secret[:,:,3] == 255)) - outside

        inside_props = []
        outside_props = []
        n = np.random.rand(len(self.circles))

        for i, [x, y, r] in enumerate(self.circles):
            x, y = int(round(x)), int(round(y))
            inside_count, outside_count = 0, 0

            for _ in range(self.num_samples):
                while True:
                    dx = np.random.uniform(-r, r)
                    dy = np.random.uniform(-r, r)
                    if (dx**2 + dy**2) <= r**2:
                        break

                x_grid = int(np.clip(np.round(x + dx), 0, self.image_size - 1))
                y_grid = int(np.clip(np.round(y + dy), 0, self.image_size - 1))
                if inside[y_grid, x_grid]:
                    inside_count += 1
                elif outside[y_grid, x_grid]:
                    outside_count += 1

            in_p  = np.clip(inside_count  / self.num_samples * (1 - (n[i] * self.noise / 100)), 0, 1)
            out_p = np.clip(outside_count / self.num_samples * (1 - (n[i] * self.noise / 100)), 0, 1)

            inside_props.append(in_p)
            outside_props.append(out_p)

        self.inside_props = inside_props
        self.outside_props = outside_props


    def __draw_plate(self):
        """
        Using generated geometry data and computed inside/outside proportions,
        draw the plate.
        """
        assert None not in [self.circles, self.inside_props, self.outside_props]

        for i, [x, y, r] in enumerate(self.circles):
            in_p, out_p = self.inside_props[i], self.outside_props[i]
            if not self.gradient:
                in_p = 1 if in_p > 0.5 else 0
                out_p = 1 - in_p

            circle_color = in_p * self.inside_color + out_p * self.outside_color
            # noise apply to the six channel, scale the entire vector
            lum_noise = np.random.normal(0, self.lum_noise)
            # only apply to vector that are on
            new_color = np.clip(circle_color + (lum_noise * (circle_color > 0)), 0, 1)
            self.__draw_ellipse([x-r, y-r, x+r, y+r], new_color)
            

    def __draw_ellipse(self, bounding_box:List, fill:ArrayLike):
        """
        Wrapper function for PIL ImageDraw. Draws to each of the
        R, G1, G2, and B channels; each channel is represented as
        a grayscale image.

        :param bounding_box: Four points to define the bounding box.
            Sequence of either [(x0, y0), (x1, y1)] or [x0, y0, x1, y1].
        :param fill: RGBOCV tuple represented as float [0, 1].
        """
        ellipse_color = (fill * 255).astype(int)
        self.channel_draws[0].ellipse(bounding_box, fill=tuple(ellipse_color[:3]), width=0)
        self.channel_draws[1].ellipse(bounding_box, fill=tuple(ellipse_color[3:]), width=0)
            
        
    def __reset_geometry(self):
        """
        Reset plate geometry. Useful if we want to regenerate the plate pattern
        with a different seed.
        """
        self.circles = None
        self.inside_props = None
        self.outside_props = None


    def __reset_images(self):
        """
        Reset plate images. Useful if we want to regenerate the plate with
        different inside/outside colors.
        """
        self.channels = [Image.new(mode='RGB', size=(self.image_size, self.image_size)) for _ in range(4)]
        self.channel_draws = [ImageDraw.Draw(ch) for ch in self.channels]


    def __reset_plate(self):
        """
        Reset geometry and images.
        """
        self.__reset_geometry()
        self.__reset_images()