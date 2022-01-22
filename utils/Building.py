class Building:
    def __init__(self, bname, zones_file=None, victims_file=None, rubbles_file=None, sizes=None, limits=None, victim_class_file=None, rubble_class_file=None):
        self.bname = bname
        self.zones_file = zones_file
        self.victims_file = victims_file
        self.rubbles_file = rubbles_file
        self.sizes = sizes
        self.limits = limits
        self.victim_class_file = victim_class_file
        self.rubble_class_file = rubble_class_file
        # self.zones_victim_file = zones_victim_file
