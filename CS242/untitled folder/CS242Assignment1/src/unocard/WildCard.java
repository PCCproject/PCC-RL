package unocard;

import game.SpecialEffectsManager;

public class WildCard extends ActionCard{
	
	private final Color color;
	private final Content content;
	private String newColor;
	private boolean usedWild;
	
	public enum Color {
        BLACK
    }
	
	public enum Content {
		WILD,
		DRAWFOURWILD
	}
	
	
	public WildCard() {
		super(ActionCard.Color.BLUE, ActionCard.Content.SKIP);
		//super(UnoCard.Color.BLACK, WildCard.Content.WILD);
		this.color = WildCard.Color.BLACK;
		this.content = WildCard.Content.WILD;
	}
	
	@Override
    public String getContent() {
    	return this.content.toString();
    }
    
    @Override
    public String getColor() {
    	return this.color.toString();
    }
	
	/**
	 * Player declares the next color to be matched (may be used on any 
	 * turn even if the player has matching color; current color may be 
	 * chosen as the next to be matched)
	 * @param effManager: the object that controls the game effects.
	 */
	@Override
	public void doSpecialEffect(SpecialEffectsManager effManager) {
		String wildCardColor = effManager.doWild();
		if (this.usedWild == false) {
			this.newColor = wildCardColor;
			this.usedWild = true;
		}
	}
	
	public void setUsedWild(String wildCardColor) {
		if (this.usedWild == false) {
			this.newColor = wildCardColor;
			this.usedWild = true;
		}
	}
	

}
