# Custom Shelf Model Spec

This is a future visual asset specification. It does not block the AI2-THOR demo.

## Shelf

- 4 vertical levels.
- 1.2 m width, 0.4 m depth, 1.8 m height.
- Product slots arranged as `level_index`, `column_index`.
- Empty slots should be representable for out-of-stock tasks.

## Retail Props

- Cart-like basket near the agent start position.
- Checkout counter with a barcode scanner zone.
- Price labels and optional barcode labels attached to product slots.

## Interactions

- Open shelf for common products.
- Closed cabinet door for hidden products.
- Sliding or hinge door reference for future custom environments.

## Metadata

- `slot_id`
- `product_id`
- `barcode`
- `expiration_date`
- `price`
- `stock_status`
