import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_clean_architecture_diagram(output_path):
    # Dimensions: 16 x 8 inches at 300 DPI
    fig, ax = plt.subplots(figsize=(16, 8.0), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.0)
    ax.axis('off')

    # Background
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Color Palette - High Contrast, Academic & Modern
    c_blue_bg = "#EFF6FF"
    c_blue_border = "#2563EB"

    c_purple_bg = "#FAF5FF"
    c_purple_border = "#7C3AED"

    c_amber_bg = "#FFFBEB"
    c_amber_border = "#D97706"

    c_green_bg = "#ECFDF5"
    c_green_border = "#059669"

    # Helper: Main Stage Card
    def draw_card(x, y, w, h, bg_color, border_color, step_num, step_title, main_title, bullet1, bullet2, tag=""):
        # Outer Card
        card = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.25",
            facecolor=bg_color, edgecolor=border_color, linewidth=2.2, zorder=2
        )
        ax.add_patch(card)

        # Step Pill Badge at Top
        badge_w = 1.7
        badge_h = 0.45
        badge_x = x + (w - badge_w) / 2
        badge_y = y + h - 0.22
        badge = patches.FancyBboxPatch(
            (badge_x, badge_y), badge_w, badge_h,
            boxstyle="round,pad=0.04,rounding_size=0.14",
            facecolor=border_color, edgecolor=border_color, zorder=3
        )
        ax.add_patch(badge)
        ax.text(x + w/2, badge_y + badge_h/2, f"STEP {step_num}",
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='#FFFFFF', zorder=4, fontfamily='DejaVu Sans')

        # Category Label
        ax.text(x + w/2, y + h - 0.65, step_title.upper(),
                ha='center', va='center', fontsize=11, fontweight='bold',
                color=border_color, zorder=4, fontfamily='DejaVu Sans')

        # Main Title (Clean, Bold, Well-Padded)
        ax.text(x + w/2, y + h - 1.25, main_title,
                ha='center', va='center', fontsize=13.5, fontweight='bold',
                color='#0F172A', zorder=4, fontfamily='DejaVu Sans')

        # Divider Line inside Card
        line = patches.FancyArrowPatch(
            (x + 0.35, y + h - 1.6), (x + w - 0.35, y + h - 1.6),
            color=border_color, linewidth=1.0, alpha=0.4, zorder=3
        )
        ax.add_patch(line)

        # Clear Bullet Points
        ax.text(x + 0.35, y + h - 2.1, f"• {bullet1}",
                ha='left', va='center', fontsize=11, color='#334155',
                zorder=4, fontfamily='DejaVu Sans')
        ax.text(x + 0.35, y + h - 2.65, f"• {bullet2}",
                ha='left', va='center', fontsize=11, color='#334155',
                zorder=4, fontfamily='DejaVu Sans')

        # Bottom Technology Pill
        if tag:
            tag_w = w - 0.6
            tag_h = 0.48
            tag_x = x + 0.3
            tag_y = y + 0.3
            tag_box = patches.FancyBboxPatch(
                (tag_x, tag_y), tag_w, tag_h,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                facecolor='#FFFFFF', edgecolor=border_color, linewidth=1.5, zorder=3
            )
            ax.add_patch(tag_box)
            ax.text(x + w/2, tag_y + tag_h/2, tag,
                    ha='center', va='center', fontsize=10.5, fontweight='bold',
                    color=border_color, zorder=4, fontfamily='DejaVu Sans')

    # Helper: Clear Connecting Arrow with Pill
    def draw_connect_arrow(x1, y1, x2, y2, label=""):
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='-|>,head_length=8,head_width=5',
            color='#1E293B', linewidth=2.5, zorder=5
        )
        ax.add_patch(arrow)
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2 + 0.36
            ax.text(mx, my, label, ha='center', va='center',
                    fontsize=10, fontweight='bold', color='#0F172A',
                    bbox=dict(boxstyle='round,pad=0.22', facecolor='#F8FAFC', edgecolor='#64748B', lw=1.2),
                    zorder=6, fontfamily='DejaVu Sans')

    # ==================== MAIN HEADER ====================
    ax.text(8.0, 7.5, "VerifyRAG: Simplified System Architecture",
            ha='center', va='center', fontsize=19, fontweight='bold', color='#0F172A', fontfamily='DejaVu Sans')
    ax.text(8.0, 7.05, "Automated claim-by-claim fact-checking pipeline to eliminate hallucinations",
            ha='center', va='center', fontsize=12, color='#64748B', fontfamily='DejaVu Sans')

    # Card dimensions
    card_w = 3.1
    card_h = 4.2
    card_y = 2.0

    # Spacing between cards: (16 - 0.6 - 4*3.1) / 3 = (16 - 0.6 - 12.4) / 3 = 3.0 / 3 = 1.0 gap!
    x1 = 0.6
    x2 = x1 + card_w + 0.95  # 0.6 + 3.1 + 0.95 = 4.65
    x3 = x2 + card_w + 0.95  # 4.65 + 3.1 + 0.95 = 8.7
    x4 = x3 + card_w + 0.95  # 8.7 + 3.1 + 0.95 = 12.75

    # ==================== 4 CORE STAGES ====================

    # STAGE 1: RETRIEVAL
    draw_card(
        x=x1, y=card_y, w=card_w, h=card_h,
        bg_color=c_blue_bg, border_color=c_blue_border,
        step_num="1", step_title="Knowledge Retrieval",
        main_title="Semantic Vector Search",
        bullet1="Splits PDF/TXT into chunks",
        bullet2="Retrieves top-4 factual passages",
        tag="FAISS + MiniLM Embeddings"
    )

    # Arrow 1 -> 2
    draw_connect_arrow(x1 + card_w, card_y + card_h/2, x2, card_y + card_h/2, label="Top Chunks")

    # STAGE 2: GENERATION & CLAIM DECOMPOSITION
    draw_card(
        x=x2, y=card_y, w=card_w, h=card_h,
        bg_color=c_purple_bg, border_color=c_purple_border,
        step_num="2", step_title="Draft & Decompose",
        main_title="LLM Draft + Claim Split",
        bullet1="LLM writes an initial answer",
        bullet2="spaCy splits into atomic claims",
        tag="Groq / OpenAI + spaCy"
    )

    # Arrow 2 -> 3
    draw_connect_arrow(x2 + card_w, card_y + card_h/2, x3, card_y + card_h/2, label="Atomic Claims")

    # STAGE 3: NLI VERIFICATION
    draw_card(
        x=x3, y=card_y, w=card_w, h=card_h,
        bg_color=c_amber_bg, border_color=c_amber_border,
        step_num="3", step_title="Fact-Checking Engine",
        main_title="Cross-Encoder NLI",
        bullet1="Entailment = Backed by source",
        bullet2="Contradiction = Hallucination",
        tag="DeBERTa-v3 Cross-Encoder"
    )

    # Arrow 3 -> 4
    draw_connect_arrow(x3 + card_w, card_y + card_h/2, x4, card_y + card_h/2, label="NLI Verdicts")

    # STAGE 4: REMEDIATION & VERIFIED OUTPUT
    draw_card(
        x=x4, y=card_y, w=card_w, h=card_h,
        bg_color=c_green_bg, border_color=c_green_border,
        step_num="4", step_title="Remediation & UI",
        main_title="Verified Answer + Citations",
        bullet1="Filters out unsupported claims",
        bullet2="Links each claim to exact chunk",
        tag="Streamlit Interactive Demo"
    )

    # ==================== BOTTOM COMPARISON BAR ====================
    comp_box = patches.FancyBboxPatch(
        (0.6, 0.45), 14.8, 1.05,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.5, zorder=2
    )
    ax.add_patch(comp_box)

    ax.text(0.9, 1.08, "PLAIN RAG (Baseline):", fontsize=11, fontweight='bold', color="#DC2626", fontfamily='DejaVu Sans')
    ax.text(3.7, 1.08, "Retrieval -> LLM writes answer -> Shown to user without checking (Unchecked Hallucinations)", 
            fontsize=10.5, color="#475569", fontfamily='DejaVu Sans')

    ax.text(0.9, 0.68, "VERIFYRAG (Proposed):", fontsize=11, fontweight='bold', color="#059669", fontfamily='DejaVu Sans')
    ax.text(3.7, 0.68, "Retrieval -> LLM Draft -> Split Claims -> DeBERTa NLI Verification -> Remediated Answer with Citations", 
            fontsize=10.5, fontweight='bold', color="#0F172A", fontfamily='DejaVu Sans')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Perfect readability architecture diagram saved to: {output_path}")

if __name__ == '__main__':
    create_clean_architecture_diagram(r"C:\Users\Vansh Gupta\.gemini\antigravity\scratch\VerifyRAG\architecture_diagram.png")
