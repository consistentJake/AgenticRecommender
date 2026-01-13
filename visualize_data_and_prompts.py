"""
Visualization script for data loading and prompt formatting.

This script demonstrates:
1. Loading data using enriched_loader
2. Creating EnrichedUser representations
3. Viewing processed data structures
4. Formatting prompts with real data
"""

import json
from pathlib import Path
from agentic_recommender.data.enriched_loader import load_singapore_data
from agentic_recommender.data.representations import EnrichedUser, CuisineProfile, CuisineRegistry
from agentic_recommender.core.prompts import PromptManager, PromptType


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def visualize_data_loading():
    """Load and visualize data from enriched_loader."""
    print_section("STEP 1: Loading Singapore Food Delivery Data")

    # Load data
    loader = load_singapore_data()

    # Show dataset statistics
    stats = loader.get_stats()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")

    # Load merged data
    merged_df = loader.load_merged()
    print(f"\nMerged DataFrame shape: {merged_df.shape}")
    print(f"Columns: {list(merged_df.columns)}")

    # Show sample data
    print("\nSample merged data (first 3 rows):")
    print(merged_df.head(3).to_string())

    return loader, merged_df


def visualize_user_representation(loader, customer_id: str = None):
    """Create and visualize EnrichedUser representation."""
    print_section("STEP 2: Creating EnrichedUser Representation")

    # Get a sample customer if not specified
    if customer_id is None:
        merged_df = loader.load_merged()
        customer_counts = merged_df['customer_id'].value_counts()
        # Get a customer with reasonable activity (5-20 orders)
        valid_customers = customer_counts[(customer_counts >= 5) & (customer_counts <= 20)]
        if len(valid_customers) > 0:
            customer_id = valid_customers.index[0]
        else:
            customer_id = customer_counts.index[0]

    print(f"Selected customer: {customer_id}")

    # Get customer orders
    customer_orders = loader.get_customer_orders(customer_id)
    print(f"\nTotal items ordered: {len(customer_orders)}")
    print(f"Number of orders: {customer_orders['order_id'].nunique()}")

    # Create EnrichedUser
    enriched_user = EnrichedUser.from_orders(customer_id, customer_orders)

    # Display user representation
    print("\n" + "-" * 80)
    print("EnrichedUser Representation:")
    print("-" * 80)

    print(f"\nCustomer ID: {enriched_user.customer_id}")
    print(f"Primary Geohash: {enriched_user.primary_geohash}")

    print("\nCuisine Distribution:")
    for cuisine, prob in sorted(enriched_user.cuisine_distribution.items(),
                                 key=lambda x: -x[1])[:5]:
        print(f"  {cuisine}: {prob:.2%}")

    print(f"\nCuisine Sequence (last 10): {enriched_user.cuisine_sequence[-10:]}")

    print("\nTemporal Patterns:")
    print(f"  Peak hours: {enriched_user.peak_hours}")
    print(f"  Peak weekdays: {enriched_user.peak_weekdays}")

    print("\nPrice Behavior:")
    print(f"  Average order total: ${enriched_user.avg_price:.2f}")
    print(f"  Price std dev: ${enriched_user.price_std:.2f}")
    print(f"  Price range: ${enriched_user.price_range[0]:.2f} - ${enriched_user.price_range[1]:.2f}")

    print("\nVendor Loyalty:")
    print(f"  Vendor diversity: {enriched_user.vendor_diversity:.2f}")
    print("  Top vendors:")
    for vendor_id, count in enriched_user.top_vendors[:3]:
        print(f"    {vendor_id}: {count} orders")

    print("\nActivity Metrics:")
    print(f"  Total orders: {enriched_user.total_orders}")
    print(f"  Total items: {enriched_user.total_items}")
    print(f"  Typical basket size: {enriched_user.typical_basket_size:.1f}")

    # Show raw dictionary format
    print("\n" + "-" * 80)
    print("EnrichedUser as Dictionary (for JSON serialization):")
    print("-" * 80)
    user_dict = enriched_user.to_dict()
    print(json.dumps(user_dict, indent=2))

    return enriched_user, customer_orders


def visualize_cuisine_profiles(loader):
    """Create and visualize CuisineProfile."""
    print_section("STEP 3: Creating Cuisine Profiles")

    # Build cuisine registry
    merged_df = loader.load_merged()
    registry = CuisineRegistry()
    registry.build_from_data(merged_df)

    # Show top cuisines
    cuisines = registry.get_all_cuisines()
    print(f"Total cuisines: {len(cuisines)}\n")

    # Show details for a few cuisines
    merged_df = loader.load_merged()
    top_cuisines = merged_df['cuisine'].value_counts().head(3)

    for cuisine in top_cuisines.index:
        profile = registry.get_profile(cuisine)
        if profile:
            print("-" * 80)
            print(f"Cuisine: {cuisine}")
            print(f"  Total orders: {profile.total_orders:,}")
            print(f"  Unique customers: {profile.unique_customers:,}")
            print(f"  Average price: ${profile.avg_price:.2f}")
            print(f"  Peak hours: {profile.peak_hours}")
            print(f"  Peak weekdays: {profile.peak_weekdays}")
            if profile.meal_time_distribution:
                print("  Meal time distribution:")
                for meal, prob in sorted(profile.meal_time_distribution.items(),
                                        key=lambda x: -x[1])[:3]:
                    print(f"    {meal}: {prob:.1%}")

    return registry


def format_order_history(customer_orders):
    """Format order history for prompt."""
    # Group by order_id and format
    orders_list = []
    for order_id in customer_orders['order_id'].unique()[:5]:  # Last 5 orders
        order_items = customer_orders[customer_orders['order_id'] == order_id]
        cuisine = order_items['cuisine'].iloc[0]
        day = order_items['day_name'].iloc[0]
        hour = order_items['hour'].iloc[0] if 'hour' in order_items.columns else 'N/A'
        total_price = order_items['unit_price'].sum()

        orders_list.append(
            f"- {day} at {hour}:00: {cuisine} (${total_price:.2f})"
        )

    return "\n".join(orders_list)


def format_candidate_product(customer_orders, loader):
    """Select and format a candidate product."""
    # Get a product the user hasn't ordered yet
    merged_df = loader.load_merged()
    user_vendors = customer_orders['vendor_id'].unique()

    # Find a vendor the user hasn't tried
    all_vendors = merged_df['vendor_id'].unique()
    new_vendors = [v for v in all_vendors if v not in user_vendors]

    if new_vendors:
        candidate_vendor = new_vendors[0]
        candidate_data = merged_df[merged_df['vendor_id'] == candidate_vendor].iloc[0]

        return f"Vendor: {candidate_vendor}\nCuisine: {candidate_data['cuisine']}\nEstimated price: ${candidate_data['unit_price']:.2f}"
    else:
        # Fallback: just use a random product
        candidate_data = merged_df.sample(1).iloc[0]
        return f"Vendor: {candidate_data['vendor_id']}\nCuisine: {candidate_data['cuisine']}\nEstimated price: ${candidate_data['unit_price']:.2f}"


def visualize_prompts(enriched_user, customer_orders, loader):
    """Visualize formatted prompts with real data."""
    print_section("STEP 4: Formatting Prompts for LLM")

    # Initialize prompt manager
    prompt_mgr = PromptManager()

    # Format data for prompts
    order_history = format_order_history(customer_orders)
    candidate_product = format_candidate_product(customer_orders, loader)

    # 1. REFLECTOR_FIRST_ROUND
    print("=" * 80)
    print("PROMPT 1: Reflector First Round (Recommendation Prediction)")
    print("=" * 80)
    print("\nPrompt Type: PromptType.REFLECTOR_FIRST_ROUND")
    print("\nFormatted Prompt:")
    print("-" * 80)

    prompt1 = prompt_mgr.render(
        PromptType.REFLECTOR_FIRST_ROUND,
        order_history=order_history,
        candidate_product=candidate_product
    )
    print(prompt1)

    # 2. ANALYST_USER
    print("\n\n" + "=" * 80)
    print("PROMPT 2: Analyst User (User Analysis)")
    print("=" * 80)
    print("\nPrompt Type: PromptType.ANALYST_USER")
    print("\nFormatted Prompt:")
    print("-" * 80)

    user_info = f"""Customer ID: {enriched_user.customer_id}
Primary Location: {enriched_user.primary_geohash}
Total Orders: {enriched_user.total_orders}
Average Order Price: ${enriched_user.avg_price:.2f}
Vendor Diversity: {enriched_user.vendor_diversity:.2f}"""

    user_history = order_history
    instruction = "Focus on sequential patterns and next-item prediction."

    prompt2 = prompt_mgr.render(
        PromptType.ANALYST_USER,
        user_info=user_info,
        user_history=user_history,
        instruction=instruction
    )
    print(prompt2)

    # 3. MANAGER_THINK
    print("\n\n" + "=" * 80)
    print("PROMPT 3: Manager Think (Reasoning)")
    print("=" * 80)
    print("\nPrompt Type: PromptType.MANAGER_THINK")
    print("\nFormatted Prompt:")
    print("-" * 80)

    context = f"""User ID: {enriched_user.customer_id}
Recent cuisine preferences: {', '.join(list(enriched_user.cuisine_distribution.keys())[:3])}
Candidate product: {candidate_product.replace(chr(10), ' | ')}"""

    scratchpad = """Previous actions:
1. Loaded user data
2. Analyzed cuisine preferences"""

    prompt3 = prompt_mgr.render(
        PromptType.MANAGER_THINK,
        context=context,
        scratchpad=scratchpad
    )
    print(prompt3)

    # 4. MANAGER_ACT
    print("\n\n" + "=" * 80)
    print("PROMPT 4: Manager Act (Action Selection)")
    print("=" * 80)
    print("\nPrompt Type: PromptType.MANAGER_ACT")
    print("\nFormatted Prompt:")
    print("-" * 80)

    prompt4 = prompt_mgr.render(
        PromptType.MANAGER_ACT,
        context=context,
        scratchpad=scratchpad + "\n3. Determined user shows strong preference for variety"
    )
    print(prompt4)

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY: Prompt Variables")
    print("=" * 80)

    print("\nAvailable Prompt Types:")
    for prompt_type in PromptType:
        try:
            template = prompt_mgr.get_template(prompt_type)
            print(f"\n{prompt_type.value}:")
            print(f"  Required variables: {template.required_vars}")
            print(f"  All variables: {template.get_variables()}")
        except ValueError:
            # Template not defined yet
            print(f"\n{prompt_type.value}: (template not defined)")


def main():
    """Main visualization function."""
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "  DATA LOADING AND PROMPT FORMATTING VISUALIZATION".center(78) + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)

    # Step 1: Load data
    loader, merged_df = visualize_data_loading()

    # Step 2: Create user representation
    enriched_user, customer_orders = visualize_user_representation(loader)

    # Step 3: Create cuisine profiles
    registry = visualize_cuisine_profiles(loader)

    # Step 4: Format and display prompts
    visualize_prompts(enriched_user, customer_orders, loader)

    print("\n\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "  VISUALIZATION COMPLETE".center(78) + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80 + "\n")

    return {
        'loader': loader,
        'enriched_user': enriched_user,
        'customer_orders': customer_orders,
        'cuisine_registry': registry
    }


if __name__ == "__main__":
    results = main()
