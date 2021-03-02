package unocard;
import java.util.ArrayList;

/**
 * This class is the list of Uno Cards in the player's hand.
 * it can draw a new card or play a card from hand.
 */
public class PlayerCard {
	protected ArrayList <UnoCard> hand;
	
	public PlayerCard() {
		this.hand = new ArrayList<UnoCard>();
	}
	
	/**
	 * Get the number of Uno Cards in the player's hand;
	 * @return number of Uno Cards in hand;
	 */
	public int getNumUnoCards() {
		return this.hand.size();
	}
	
	/**
	 * add a Uno Card to the player's hand
	 * @param newCard: the Uno Card to be added
	 */
	public void addCard(UnoCard newCard) {
		this.hand.add(newCard);
	}
	
	/**
	 * get a Uno Card in the player's hand using the given index, and return the name in String
	 * @param idx: index given
	 */
	public String getCard(int idx) {
		try{
			return this.hand.get(idx).toString();
		} catch(IndexOutOfBoundsException e){
			return null;
		}
	}
	
	/**
	 * play the Uno Card with the given index
	 * @param idx: the index of the card in the player's hand
	 * @return the Uno Card played
	 */
	public UnoCard playCard(int idx) {
		try {
			UnoCard card = this.hand.remove(idx);
			return card;
		}catch(IndexOutOfBoundsException e) {
			e.printStackTrace(); 
			return null;
		}
	}
	
	
}
