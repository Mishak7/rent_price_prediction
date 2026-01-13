from pydantic import BaseModel, Field
from typing import Optional

class FlatRawInfo(BaseModel):
    listing_id: int = Field(description="ID объявления")
    total_area: Optional[float] = Field(default= 60, description="Общая площадь квартиры")
    rooms_count: Optional[int] = Field(default=2, description="Количество комнат")
    renovation: Optional[str] = Field(default="unknown", description="Дизайнерский/Евроремонт/Косметический/Без ремонта")
    parking: Optional[str] = Field(default="unknown", description="Наземная/надземная")
    lat: Optional[float] = Field(None, description="lat")
    lon: Optional[float] = Field(None, description="lon")
    building_type: Optional[str] = Field(default="unknown",
                                         description="Кирпичный/Монолитно-кирпичный/Монолитный/Панельный/Блочный/Старый фонд")
    room_type: Optional[str] = Field(default="unknown", description="Комнаты: Изолированная/Смежная/Оба варианта")
    loggia_count: Optional[int] = Field(None, description="Количество лоджий")
    floor: Optional[int] = Field(None, description="Этаж")
    floors_total: Optional[int] = Field(None, description="Этажность")
    city: Optional[str] = Field(None, description="Город")
    street: Optional[str] = Field(None, description="Название улицы. Например: Ленинский проспект")
    complex: Optional[str] = Field(None, description="Название ЖК. Например: Twin Peaks")
    description: Optional[str] = Field(None, description="Описание")