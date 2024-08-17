# parking_data_simulation.py

import sqlite3
import os
import time
from datetime import datetime

class ParkingDataSimulation:
    def __init__(self, db_file):
        self.db_file = db_file
        self.parking_areas = {
            "B3": {"B3 A 区": 50, "B3 B 区": 60, "B3 C 区": 70},
            "B4": {"B4 A 区": 100, "B4 B 区": 80, "B4 C 区": 90, "B4 D 区": 120}
        }
        self.init_db()

    # 初始化数据库
    def init_db(self):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parking_data (
                timestamp TEXT,
                level TEXT,
                area TEXT,
                available_spots INTEGER,
                total_spots INTEGER,
                PRIMARY KEY (timestamp, level, area)
            )
        ''')
        conn.commit()
        conn.close()

    # 基于时间和节日的停车位占用率
    def calculate_occupancy_rate(self, hour, is_weekend, is_holiday):
        if is_holiday:
            return 0.9
        elif is_weekend:
            if 10 <= hour < 18:
                return 0.75
            else:
                return 0.5
        else:
            if 8 <= hour < 10 or 17 <= hour < 20:
                return 0.7
            elif 12 <= hour < 14:
                return 0.6
            else:
                return 0.3

    # 模拟车位状态更新
    def update_parking_slots(self):
        parking_data = []
        now = datetime.now()
        hour = now.hour
        is_weekend = now.weekday() >= 5
        is_holiday = now.month == 12 and now.day == 25

        for level, areas in self.parking_areas.items():
            for area, total_slots in areas.items():
                occupancy_rate = self.calculate_occupancy_rate(hour, is_weekend, is_holiday)
                occupied_spots = int(total_slots * occupancy_rate)
                available_spots = total_slots - occupied_spots

                parking_data.append({
                    "timestamp": now.strftime('%Y-%m-%d %H:%M:%S'),
                    "level": level,
                    "area": area,
                    "available_spots": available_spots,
                    "total_spots": total_slots,
                })
        
        return parking_data

    # 存储数据到数据库
    def store_data(self, data):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        for entry in data:
            cursor.execute('''
                INSERT OR REPLACE INTO parking_data (timestamp, level, area, available_spots, total_spots)
                VALUES (?, ?, ?, ?, ?)
            ''', (entry['timestamp'], entry['level'], entry['area'], entry['available_spots'], entry['total_spots']))

        conn.commit()
        conn.close()

    # 实时数据生成器
    def generate_real_time_parking_data(self):
        while True:
            parking_data = self.update_parking_slots()
            self.store_data(parking_data)
            time.sleep(60)  # 每60秒更新一次数据

    def start_simulation(self):
        self.generate_real_time_parking_data()

