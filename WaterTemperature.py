from datetime import datetime, timedelta
import pandas as pd
import geemap
import ee
ee.Initialize()


def generate_dates(start_date, end_date):
    """
    Generate all dates between two given dates.

    Parameters:
    start_date (str): The start date in 'YYYY-MM-DD' format.
    end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
    list: A list of dates between start_date and end_date.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    date_list = []
    current_date = start

    while current_date <= end:
        date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    return date_list


class WaterTemperature(object):

    def __init__(self):
        self.lat = None
        self.lon = None
        self.TempTimeSeries = None
        self.roi = None
        self.sat = 'ECMWF/ERA5_LAND/DAILY_AGGR'
        self.reducer = ee.Reducer.mean()
        self.scale = 1113  # meters
        self.labels = ['lake_mix_layer_temperature',
                       'lake_mix_layer_depth_min',
                       'lake_mix_layer_depth_max']

    def setROI(self, radio):
        """
        Sets the region of interest (ROI) with a given radius.

        Parameters
        ----------
        radio : float
            The radius of the ROI in meters.
        """
        if radio < self.scale:
            print("Region is too small")
            return
        else:
            self.roi = ee.Geometry.Point([self.lon, self.lat]).buffer(radio)

    def getTempTime(self, date):
        """
        Retrieves Water Temperature data for a specific date.

        Parameters
        ----------
        date : str
            The date in 'YYYY-MM-DD' format.

        Returns
        -------
        dict
            A dictionary with wind speed components for the specified date and hour.
        """
        if self.roi == None:
            print("Set a valid Region")
            return
        gDate = ee.Date(date)
        self.gDate = gDate
        image = ee.ImageCollection(self.sat)\
            .filterBounds(self.roi)\
            .filterDate(gDate, gDate.advance(1, 'day'))\
            .select(self.labels)
        imageReduce = image.first().reduceRegion(
            reducer=self.reducer,
            geometry=self.roi,
            scale=self.scale,
            maxPixels=1e9
        )
        return imageReduce.getInfo()

    def getWaterTempDates(self, start_date, end_date):
        """
        Retrieves water temperature data for all dates between the start and end date.

        Parameters
        ----------
        start_date : str
            The start date in 'YYYY-MM-DD' format.
        end_date : str
            The end date in 'YYYY-MM-DD' format.

        Returns
        -------
        pandas.DataFrame
        """
        Dates = generate_dates(start_date, end_date)
        data = {'Date': []}
        for label in self.labels:
            data[label] = []
        for date in Dates:
            print("Processing...", date)
            Tempdata = self.getTempTime(date)
            for label in self.labels:
                if label == 'lake_mix_layer_temperature':
                    data[label].append(
                        Tempdata[label]-273.15)
                else:
                    data[label].append(
                        Tempdata[label])
            data['Date'].append(date)
        return pd.DataFrame(data)


lakeboundary = {'PIGEON LAKE': './BoundaryLake/PigeonLake/pige_bdy_py_tm.shp',
                'CHESTERMERE LAKE': './BoundaryLake/ChestermereLake/ches_bdy_py_tm.shp',
                'MINNIE LAKE': './BoundaryLake/MinnieLake/minn_bdy_py_tm.shp'}

# path = './DIG_2008_0828/'
# name = 'pige_bdy_py_tm.shp'

years = ['2019', '2020', '2021', '2022', '2023']
pathData = "./ERA5-Land/"
for lake in lakeboundary:
    watertemperature = WaterTemperature()
    watertemperature.roi = geemap.shp_to_ee(lakeboundary[lake])
    for year in years:
        df = watertemperature.getWaterTempDates(year+"-05-01", year+"-09-30")
        df.to_csv(pathData+year+lake+"WaterTemperature.csv")
