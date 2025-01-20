mkdir -p ./chromatic-gamut-grid-outputs

# 3D 
python chromatic-gamut.py --dimension 3 --display_type ideal --primary_wavelengths 425 515 695  --output_file ./chromatic-gamut-grid-outputs/3d-ideal-simplex.mp4 --fps 30 --total_frames 90
python chromatic-gamut.py --dimension 3 --display_type viz-efficient --primary_wavelengths 425 515 695 --viz_efficient_wavelengths 450 535 600   --output_file ./chromatic-gamut-grid-outputs/3d-viz_eff.mp4 --fps 30 --total_frames 90
python chromatic-gamut.py --dimension 3 --display_type ours --primary_wavelengths 425 515 695  --output_file ./chromatic-gamut-grid-outputs/3d-our-gamut.mp4 --fps 30 --total_frames 90

# 4D
python chromatic-gamut.py --dimension 4 --template govardovskii --display_type ideal --primary_wavelengths 410 510 585 695 --output_file ./chromatic-gamut-grid-outputs/4d-ideal-simplex.mp4 --fps 30 --total_frames 90
python chromatic-gamut.py --dimension 4 --template govardovskii --display_type viz-efficient  --primary_wavelengths 410 510 585 695 --viz_efficient_wavelengths 445 530 585 630  --output_file ./chromatic-gamut-grid-outputs/4d-viz_eff.mp4 --fps 30 --total_frames 90
python chromatic-gamut.py --dimension 4 --template govardovskii --display_type ours  --primary_wavelengths 410 510 585 695  --output_file ./chromatic-gamut-grid-outputs/4d-our-gamut.mp4 --fps 30 --total_frames 90