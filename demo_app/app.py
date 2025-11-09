from flask import Flask, render_template, request

app = Flask(__name__)

PRODUCTS = [
    {"id": 1, "name": "Widget A", "price": 19.99, "rating": 4.3},
    {"id": 2, "name": "Widget B", "price": 9.99, "rating": 3.8},
    {"id": 3, "name": "Gadget C", "price": 24.50, "rating": 4.7},
    {"id": 4, "name": "Gadget D", "price": 14.25, "rating": 4.0},
]


def get_variant():
    v = request.args.get("variant", "A").upper()
    return "B" if v == "B" else "A"


@app.route("/")
def index():
    variant = get_variant()
    cta_text = "Get Started" if variant == "A" else "Start Free Trial"
    show_incidents_chip = variant == "B"
    return render_template(
        "index.html",
        variant=variant,
        cta_text=cta_text,
        show_incidents_chip=show_incidents_chip,
        title="Home",
    )


@app.route("/products")
def products():
    variant = get_variant()
    show_rating_column = variant == "B"
    default_sort = "price" if variant == "A" else "rating"
    products = sorted(
        PRODUCTS,
        key=lambda x: x[default_sort],
        reverse=(default_sort == "rating"),
    )
    return render_template(
        "products.html",
        variant=variant,
        products=products,
        show_rating_column=show_rating_column,
        default_sort=default_sort,
        title="Products",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
