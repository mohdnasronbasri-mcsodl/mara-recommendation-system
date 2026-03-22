# ========================================
# PROGRAM RECOMMENDATIONS WITH FIXED DISPLAY
# ========================================
st.markdown("### 🎯 Program Recommendations")
st.caption("Click on any program to see detailed score breakdown")

eligible_count = len([p for p in all_programs_with_scores if p['eligible']])
st.caption(f"📊 Showing {eligible_count} eligible programs out of {len(all_programs_with_scores)} total")

# Display recommendations with expanders
for i, prog in enumerate(all_programs_with_scores, 1):
    if prog['eligible']:
        star = "⭐ " if prog['in_original'] else ""
        
        # Use empty expander, then add custom header inside
        with st.expander(""):
            # Custom header with name left, scores right
            st.markdown(f"""
            <div style='display: flex; justify-content: space-between; align-items: center; width: 100%; margin-bottom: 10px;'>
                <span style='font-weight: bold; font-size: 1em;'>{i}. {star}{prog['name']}</span>
                <div style='text-align: right;'>
                    <span style='color: #1e88e5; font-weight: bold;'>🎯 {prog['total_score']}%</span><br>
                    <span style='font-size: 0.75em; color: #666;'>📊 Eligibility: {prog['academic_score']}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Determine suitability level based on TOTAL score
            if prog['total_score'] >= 80:
                level_emoji = "🟢"
                level_text = "Highly Suitable"
            elif prog['total_score'] >= 60:
                level_emoji = "🟡"
                level_text = "Moderately Suitable"
            else:
                level_emoji = "🔴"
                level_text = "Less Suitable"
            
            # Display status
            st.markdown(f"""
            <div style='margin-bottom: 15px;'>
                <span style='font-size: 1.1em; font-weight: bold;'>{level_emoji} {level_text}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Get detailed breakdown
            detailed = prog['detailed']
            
            # Formula explanation
            st.markdown(f"""
            <div style='margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 8px;'>
                <p style='margin: 5px 0 0 0; font-size: 0.85em;'><b>Formula:</b> {detailed['weight_formula']}</p>
                <p style='margin: 2px 0 0 0; font-size: 0.75em; color: #666;'>{detailed['formula_explanation']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Three columns for the three components
            col_a, col_b, col_c = st.columns(3)
            
            # Column 1: Academic Performance (80%)
            with col_a:
                st.markdown(f"""
                <div style='background-color: #e8f4fd; padding: 10px; border-radius: 8px; height: 100%;'>
                    <h4 style='margin: 0 0 8px 0;'>📚 Academic</h4>
                    <p style='font-size: 1.5em; font-weight: bold; margin: 0; color: #1e88e5;'>{prog['academic_score']}%</p>
                    <p style='font-size: 0.7em; color: #666; margin-bottom: 8px;'>Weight: 80%</p>
                    <hr style='margin: 8px 0;'>
                """, unsafe_allow_html=True)
                
                # Show subject breakdown
                for subj in detailed['breakdown']['academic']['subjects']:
                    if subj['grade_value'] > 0:
                        st.markdown(f"""
                        <div style='display: flex; justify-content: space-between; font-size: 0.75em; margin-bottom: 4px;'>
                            <span><b>{subj['subject']}</b> ({subj['grade']})</span>
                            <span><b>{subj['contribution']:.1f}</b> (×{subj['weight']})</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <hr style='margin: 8px 0;'>
                    <div style='font-size: 0.65em; color: #666;'>{detailed['breakdown']['academic']['calculation']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Column 2: Demographic Context (10%)
            with col_b:
                st.markdown(f"""
                <div style='background-color: #fef4e8; padding: 10px; border-radius: 8px; height: 100%;'>
                    <h4 style='margin: 0 0 8px 0;'>🏠 Demographic</h4>
                    <p style='font-size: 1.5em; font-weight: bold; margin: 0; color: #fb8c00;'>{prog['demographic_score']}%</p>
                    <p style='font-size: 0.7em; color: #666; margin-bottom: 8px;'>Weight: 10%</p>
                    <hr style='margin: 8px 0;'>
                """, unsafe_allow_html=True)
                
                for comp in detailed['breakdown']['demographic']['components']:
                    st.markdown(f"""
                    <div style='margin-bottom: 8px;'>
                        <div style='display: flex; justify-content: space-between; font-size: 0.75em;'>
                            <span><b>{comp['factor']}:</b></span>
                            <span><b>{comp['score']}</b> / {comp['max_score']}</span>
                        </div>
                        <div style='font-size: 0.65em; color: #666;'>{comp['note']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='font-size: 0.65em; color: #666; margin-top: 8px;'>
                        {detailed['breakdown']['demographic']['calculation']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Column 3: Preference Alignment (10%)
            with col_c:
                pref_bonus = prog['preference_bonus']
                if pref_bonus > 0:
                    bg_color = "#e8f5e9"
                    border_color = "#4caf50"
                else:
                    bg_color = "#fff3e0"
                    border_color = "#ff9800"
                
                st.markdown(f"""
                <div style='background-color: {bg_color}; padding: 10px; border-radius: 8px; border-left: 4px solid {border_color}; height: 100%;'>
                    <h4 style='margin: 0 0 8px 0;'>⭐ Preference</h4>
                    <p style='font-size: 1.5em; font-weight: bold; margin: 0; color: {border_color};'>{pref_bonus} / 15</p>
                    <p style='font-size: 0.7em; color: #666; margin-bottom: 8px;'>Weight: 10% (max 15 points)</p>
                    <hr style='margin: 8px 0;'>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='font-size: 0.75em; margin-bottom: 8px;'>
                        {detailed['breakdown']['preference']['note']}
                    </div>
                """, unsafe_allow_html=True)
                
                if detailed['breakdown']['preference']['original_choices']:
                    st.markdown("**Your original choices:**")
                    for j, choice in enumerate(detailed['breakdown']['preference']['original_choices'], 1):
                        st.markdown(f"&nbsp;&nbsp;{j}. {choice[:50]}...")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Progress bar showing total composition
            st.markdown("---")
            st.markdown("### 📊 Score Composition")
            
            # Calculate contributions
            academic_contrib = prog['academic_score'] * 0.8
            demo_contrib = prog['demographic_score'] * 0.1
            pref_contrib = prog['preference_bonus']
            
            # Simple bar using HTML/CSS
            st.markdown(f"""
            <div style='margin: 10px 0;'>
                <div style='display: flex; height: 30px; border-radius: 5px; overflow: hidden;'>
                    <div style='background-color: #1e88e5; width: {academic_contrib}%; text-align: center; color: white; font-weight: bold; font-size: 0.75em;'>Academic {academic_contrib:.1f}%</div>
                    <div style='background-color: #fb8c00; width: {demo_contrib}%; text-align: center; color: white; font-weight: bold; font-size: 0.75em;'>Demo {demo_contrib:.1f}%</div>
                    <div style='background-color: #4caf50; width: {pref_contrib}%; text-align: center; color: white; font-weight: bold; font-size: 0.75em;'>Pref {pref_contrib:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"Total: {prog['total_score']}% = Academic ({prog['academic_score']}% × 0.8) + Demographic ({prog['demographic_score']}% × 0.1) + Preference Bonus ({pref_bonus})")
            
            # Brief explanation
            st.info(f"💡 **Why this score?** {prog['explanation']}")
    
    else:
        # Not eligible
        st.markdown(f"""
        <div style='margin-bottom: 8px; padding: 8px; border-left: 5px solid #dc3545; border-radius: 5px; background-color: #ffffff; border: 1px solid #e0e0e0;'>
            <span style='font-size: 0.9em; color: #000000;'><b>{i}. {prog['name']}</b></span><br>
            <span style='font-size: 0.75em; color: #dc3545;'><b>❌ Not eligible:</b> {prog['reason']}</span>
        </div>
        """, unsafe_allow_html=True)
