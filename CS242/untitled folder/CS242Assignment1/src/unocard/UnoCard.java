package unocard;

import game.SpecialEffectsManager;

/**
 * This is the superclass for all Uno Cards. 
 * All Uno Cards have color and contents, where color is one of five colors,
 * and content is either a number from 0 to 9 or an effect.
 */
public abstract class UnoCard {
	
	/**
	 * This enum represents a type whose elements are colors of the uno card.
	 */
	public enum Color {
		BLUE,
		RED,
		GREEN,
		YELLOW,
		BLACK
	}
	
	/**
	 * Get Uno Card content. Either a number or an effect.
	 * @return a string of the content of the card.
	 */
	public abstract String getContent(); 
	
	/**
	 * Get Uno Card color.
	 * @return a string of the color of the card.
	 */
	public abstract String getColor();
	
	/**
	 * Check whether the Uno Card match with another in color, number, or symbol.
	 * Wild Cards match with cards of any color.
	 * @param ucard: the other Uno Card.
	 * @return a boolean that is true if cards match or false if not.
	 */
	public abstract boolean doMatch(UnoCard ucard);
	
	
	public abstract void doSpecialEffect(SpecialEffectsManager effManager);
	
	
	@Override
	public String toString() {
		String ret = this.getColor() + this.getContent();
		return ret;
	}
	
	
}
	
	
