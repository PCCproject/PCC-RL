package unocard;
import game.SpecialEffectsManager;

public class DrawFourWildCard extends WildCard{
	
	private final Color color;
	private final Content content;
	
	public enum Color {
        BLACK
    }
	
	public enum Content {
		WILD,
		DRAWFOURWILD
	}
	
	
	public DrawFourWildCard() {
		super();
		this.color = DrawFourWildCard.Color.BLACK;
		this.content = DrawFourWildCard.Content.DRAWFOURWILD;
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
	 * Player declares the next color to be matched; the next player in sequence 
	 * draws four cards and misses a turn unless they have another Draw Four Wild 
	 * card to "stack" the number of cards to draw for the next player 
	 * @param effManager: the object that controls the game effects.
	 */
	@Override
	public void doSpecialEffect(SpecialEffectsManager effManager) {
		effManager.doDrawFourWild();
	
		String wildColor = effManager.doWild();
		super.setUsedWild(wildColor);
		
	}
	
}
