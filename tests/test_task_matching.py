from retail_thor.generator import select_candidates


CATALOG = [
    {
        "product_id": "apple",
        "category": "fruit",
        "attributes": ["fresh", "healthy"],
        "price": 6,
        "pickupable": True,
    },
    {
        "product_id": "bread",
        "category": "bakery",
        "attributes": ["breakfast", "staple_food"],
        "price": 12,
        "pickupable": True,
    },
    {
        "product_id": "bottle",
        "category": "drink",
        "attributes": ["bottled", "cold_drink"],
        "price": 4,
        "pickupable": True,
    },
]


def test_select_candidates_filters_by_structured_constraints():
    assert [p["product_id"] for p in select_candidates(CATALOG, {"category": "drink"})] == ["bottle"]
    assert [p["product_id"] for p in select_candidates(CATALOG, {"attribute": "breakfast"})] == ["bread"]
    assert [p["product_id"] for p in select_candidates(CATALOG, {"max_price": 5})] == ["bottle"]
    assert [p["product_id"] for p in select_candidates(CATALOG, {"category": "fruit", "attribute": "healthy"})] == [
        "apple"
    ]
