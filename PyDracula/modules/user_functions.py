import os

import pandas as pd
import folium
import base64

from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QListWidgetItem
from PySide6.QtGui import QPixmap, QColor
from PySide6.QtCore import Qt, QUrl
from PySide6.QtWebEngineCore import QWebEngineSettings


class CustomListItem(QWidget):
    def __init__(self, item_data, parent=None):
        super().__init__(parent)
        self.item_data = item_data
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5) # 위젯 내부 여백

        # 이미지 라벨
        image_label = QLabel()
        image_label.setFixedSize(60, 60)

        image_path = os.path.join(os.getcwd(), "init", "data", "Images", self.item_data.get("image_path"))
        if image_path:
            pixmap = QPixmap(image_path)
        else:
            # 이미지가 없을 경우, 이름에 따라 색상 생성
            pixmap = QPixmap(60, 60)
            color = QColor((hash(self.item_data['name']) & 0x00FFFFFF) | 0x00A0A0A0)
            pixmap.fill(color)
            
        image_label.setPixmap(pixmap.scaled(
            image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        
        # 텍스트 정보 (이름, 위도, 경도)
        text_info_layout = QVBoxLayout()
        name_label = QLabel(self.item_data["name"])
        name_label.setStyleSheet("font-weight: bold;")
        
        coords_label = QLabel(f"위도: {self.item_data['latitude']:.4f}, 경도: {self.item_data['longitude']:.4f}")
        coords_label.setStyleSheet("font-size: 9pt; color: gray;")

        text_info_layout.addWidget(name_label)
        text_info_layout.addWidget(coords_label)
        text_info_layout.addStretch()

        layout.addWidget(image_label)
        layout.addLayout(text_info_layout)


class UserClass():
    def __init__(self, ui, main_window):
        # STORE WIDGETS AND MAIN WINDOW
        self.ui = ui
        self.main = main_window

        # VARIABLES
        self.excel_path = os.path.join(os.getcwd(), "init", "data", "data.xlsx")
        self.html_path_total = os.path.join(os.getcwd(), "init", "map", "total_damagemap.html")
        self.html_path_each = os.path.join(os.getcwd(), "init", "map", "each_damagemap.html")
        self.damage_data = []

        # INITIALIZE
        settings1 = self.ui.webEngineView.settings()
        settings1.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings1.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)

        self.ui.DamagelistWidget.itemClicked.connect(self.ListViewSelected)
        self.ui.InitializeButton.clicked.connect(self.InitializeBtnClicked)

        self.InitializeProject()

        # INITIALIZATION COMPLETE
        print("UserClass Initialized")

    # RECEIVE PROJECT/USER INPUT
    # ///////////////////////////////////////////////////////////////
    def InitializeProject(self):
        # 1. Read Excel File from the Project
        loaded_df = self.readExcel()

        # 2. Make Damage List and Visualization
        self.damage_data = self.makeDamageList(loaded_df)
        self.showDamageList()

        # 3. Make Damage Map and Visualization
        if self.damage_data and len(self.damage_data) > 0:
            self.makeEachDamageMap(self.damage_data[0])  # First Item
            self.makeDamageMap()
            self.showDamageMap(mode="total")

    def ListViewSelected(self, current_item):
        # 1. Read Data from the Selected Item
        if current_item is None:
            return
        item_data = current_item.data(Qt.ItemDataRole.UserRole)

        # 2. Make Individual Map with the Data
        self.makeEachDamageMap(item_data)

        # 3. Show the Individual Map
        self.showDamageMap(mode="each")
    
    def InitializeBtnClicked(self):
        self.showDamageMap(mode="total")
    
    # PROCESS FUNCTIONS
    # ///////////////////////////////////////////////////////////////
    def readExcel(self):
        df = pd.read_excel(self.excel_path)
        return df

    def makeDamageList(self, df):
        if df is None or df.empty:
            print("Invalid DataFrame")
            return
        
        ITEMS_DATA = []

        for index, row in df.iterrows():
            item_dict = {
                "name": f"항목 {index}",  
                "image_path": row['이미지 경로'], 
                "latitude": row['위도'],
                "longitude": row['경도']
            }
            ITEMS_DATA.append(item_dict)
        
        return ITEMS_DATA
    
    def showDamageList(self):
        if self.damage_data is None or len(self.damage_data) == 0:
            self.ui.DamagelistWidget.clear()
            print("No Damage Data to Show")
            return
        
        self.ui.DamagelistWidget.clear()

        for item in self.damage_data:
            list_item = QListWidgetItem(self.ui.DamagelistWidget)
            custom_widget = CustomListItem(item)
            list_item.setSizeHint(custom_widget.sizeHint())
            list_item.setData(Qt.ItemDataRole.UserRole, item)  # Store item data
            self.ui.DamagelistWidget.addItem(list_item)
            self.ui.DamagelistWidget.setItemWidget(list_item, custom_widget)

    def makeDamageMap(self):
        if self.damage_data is None or len(self.damage_data) == 0:
            print("Invalid Damage Data")
            return
        
        # 1. Calculate Center of the Map
        avg_lat = sum(item['latitude'] for item in self.damage_data) / len(self.damage_data)
        avg_lon = sum(item['longitude'] for item in self.damage_data) / len(self.damage_data)
        map_center = [avg_lat, avg_lon]

        # 2. Create Folium Map
        damage_map = folium.Map(location=map_center, zoom_start=11)

        # 3. Add Markers to the Map
        for item in self.damage_data:
            # 3.1. Encode image to base64
            image_path = os.path.join(os.getcwd(), "init", "data", "Images", item.get("image_path"))
            if os.path.exists(image_path):
                encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode()
            else:
                encoded_image = None
            
            # 3.2. Create Popup HTML
            popup_html = self.makePopupHtml(item, encoded_image)

            # 3.3. Add Marker
            if encoded_image is None:
                popup = folium.Popup(popup_html, max_width=240)
            else:
                iframe = folium.IFrame(popup_html, width=240, height=280)
                popup = folium.Popup(iframe)
            
            folium.Marker(
                location=[item['latitude'], item['longitude']],
                popup=popup,
                tooltip=item['name']
            ).add_to(damage_map)

        # 4. Save Map to HTML
        damage_map.save(self.html_path_total)
    
    def makeEachDamageMap(self, map_data):
        if map_data is None:
            print("Invalid Damage Data")
            return
        
        # 1. Calculate Center of the Map
        map_center = [map_data['latitude'], map_data['longitude']]

        # 2. Create Folium Map
        damage_map = folium.Map(location=map_center, zoom_start=15)

        # 3. Add Markers to the Map
        # 3.1. Encode image to base64
        image_path = os.path.join(os.getcwd(), "init", "data", "Images", map_data.get("image_path"))
        if os.path.exists(image_path):
            encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode()
        else:
            encoded_image = None

        # 3.2. Create Popup HTML
        popup_html = self.makePopupHtml(map_data, encoded_image)

        # 3.3. Add Marker
        if encoded_image is None:
            popup = folium.Popup(popup_html, max_width=240)
        else:
            iframe = folium.IFrame(popup_html, width=240, height=280)
            popup = folium.Popup(iframe)

        folium.Marker(
            location=[map_data['latitude'], map_data['longitude']],
            popup=popup,
            tooltip=map_data['name']
            ).add_to(damage_map)

        # 4. Save Map to HTML
        damage_map.save(self.html_path_each)
    
    def showDamageMap(self, mode="total"):
        # 1. Read Mode and Select HTML Path
        if mode == "total":
            current_html = self.html_path_total
        elif mode == "each":
            current_html = self.html_path_each
        else:
            print("Invalid mode for DamageMap")
            return
        
        # 2. Load HTML to the QWebEngineView
        self.ui.webEngineView.load(QUrl.fromLocalFile(current_html))
    
    # UTILS
    # ///////////////////////////////////////////////////////////////
    def makePopupHtml(self, item, encoded_image=None):
        if item is None:
            return ""
        
        if encoded_image is None:
            html_template = f"<b>{item['name']}</b><br>위도: {item['latitude']}<br>경도: {item['longitude']}<br>(이미지 없음)"
        else:
            html_template = f"""
        <b>{item['name']}</b><br>
        위도: {item['latitude']:.6f}<br>
        경도: {item['longitude']:.6f}<br><br>
        <img src="data:image/jpeg;base64,{encoded_image}" width="200">
        """
        return html_template