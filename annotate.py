from ultralytics import solutions

# Angiv stien til den video- eller billedfil, du vil annotere
VIDEO_PATH = 'parkingvideo.mp4'

# 1. Initialiser klassen UDEN argumenter
annotator = solutions.ParkingPtsSelection()

# 2. Start annoterings-GUI'en ved at kalde .run() og give den stien som argument
annotator.run(source=VIDEO_PATH)

# Outputtet gemmes stadig som 'bounding_boxes.json' i din projektmappe.